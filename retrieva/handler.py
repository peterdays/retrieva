import logging
import sys
from typing import Optional

import weaviate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import (PromptTemplate, Settings, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.openai import (OpenAIEmbedding,
                                           OpenAIEmbeddingModelType)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from retrieva import LOGGER, ROOT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class RagHandler():
    def __init__(self, index_name: str, data_path: str,
                 weaviate_url: str, cloud_based: bool = False,
                 num_workers_injection: Optional[int] = None) -> None:
        """
        Handler class to setup the RAG pipeline. Exposes a method to strem the
        response to a query. Currently implements a local (based on
        Hugging face models) and a remote approach (based on the openAI API)

        :param index_name: name of the collection in the weaviate db
        :param data_path: path to the files to be injected
        :param weaviate_url: weaviate db url
        :param local_pipeline: whether the RAG system is based on cloud
        (OpenAI) or local models, defaults to False
        :param num_workers_injection: number os workers used in the injection
        """
        self.data_path = data_path
        self.cloud_based = cloud_based

        # generating the index
        documents = SimpleDirectoryReader(data_path).load_data()

        LOGGER.info("Using weaviate db at %s", weaviate_url)
        client = weaviate.Client(weaviate_url)
        # load the collection or not
        index_exists = client.schema.exists(index_name)

        # TODO: make it easier to include other pipelines
        if cloud_based:
            transformations = self._use_openai_pipeline()
        else:
            transformations = self._use_huggingfaces_pipeline()

        # creating the index vs loading it
        if index_exists:
            LOGGER.info("Loading %s...", index_name)
            vector_store = WeaviateVectorStore(
                weaviate_client=client, index_name=index_name
            )
            index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            # If you want to load the index later, be sure to give it a name!
            vector_store = WeaviateVectorStore(
                weaviate_client=client, index_name=index_name
            )
            LOGGER.info("Creating new index from %s", self.data_path)

            pipeline = IngestionPipeline(
                transformations=transformations,
                vector_store=vector_store,  # directly injecting in the db
                docstore=SimpleDocumentStore()  # looking for duplicate documents
            )

            # Ingest directly into a vector db
            pipeline.run(documents=documents, show_progress=True,
                        num_workers=num_workers_injection)

            index = VectorStoreIndex.from_vector_store(vector_store)

        # index
        self.index = index

    def user_prompt_streaming(self, prompt: str, similarity: int = 2):

        query_engine = self.index.as_query_engine(
            streaming=True,
            similarity_top_k=similarity,
            response_mode="tree_summarize" if self.cloud_based else "compact",
            verbose=True
        )

        if self.cloud_based:
            # TODO: parametrize the promp update
            self._update_openai_summary_prompt(query_engine)

        self.engine = query_engine
        response_stream = query_engine.query(prompt)

        return response_stream

    def _use_openai_pipeline(self):
        # define underlying LLM
        Settings.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        # use in a pipeline
        llm = OpenAI(model="gpt-3.5-turbo")
        embed_model = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
        )
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=15),
            TitleExtractor(nodes=5, llm=llm),
            embed_model
        ]
        Settings.embed_model = embed_model

        return transformations

    def _update_openai_summary_prompt(self, engine: BaseQueryEngine):
        # updating the tree summarize prompt according to business requirements

        new_summary_tmpl_str = (
            "You are a summarization service to help users navigate "
            "proprietary documentation of their companies.\n"
            "ALWAYS include the filepaths from the nodes in the context "
            "information to support the answer as well as further reading "
            "that may be mentioned in the docs themselves. "
            f"Remove '{ROOT_PATH}' from the metadatas' filepaths.\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "If the query and the context don't make sense answer with "
            "'Warning: There is no good match in the docs for this prompt!'.\n"
            "Given the context information and not prior knowledge, "
            "answer the query. Don't forget to cite the original docs.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

        engine.update_prompts(
            {"response_synthesizer:summary_template": new_summary_tmpl}
        )

    def _use_huggingfaces_pipeline(self):
        llm = Ollama(model="zephyr-local", request_timeout=60.0)

        Settings.llm = llm

        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=10),
            embed_model
        ]
        Settings.embed_model = embed_model

        return transformations

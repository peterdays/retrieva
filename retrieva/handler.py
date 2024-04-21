import logging
import sys
import weaviate
from llama_index.core import (PromptTemplate, Settings, SimpleDirectoryReader,
                              StorageContext, VectorStoreIndex)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from typing import Optional
from retrieva.data import DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class RagHandler():
    def __init__(self, index_name: str, data_path: str  = DATA_PATH,
                 weaviate_url: Optional[str] = None
                 ) -> None:
        # generating the index
        documents = SimpleDirectoryReader(data_path).load_data()

        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=15)
        # global
        Settings.text_splitter = text_splitter

        storage_context = None
        if weaviate_url:
            client = weaviate.Client(weaviate_url)
            # If you want to load the index later, be sure to give it a name!
            vector_store = WeaviateVectorStore(
                weaviate_client=client, index_name=index_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # index
        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[text_splitter],
            storage_context=storage_context
        )

    def user_prompt_streaming(self, prompt: str, similarity: int = 2,
                              response_mode: str = "tree_summarize"):
        # set Logging to DEBUG for more detailed outputs
        query_engine = self.index.as_query_engine(streaming=True,
                                                  similarity_top_k=similarity,
                                                  response_mode=response_mode)
        if response_mode == "tree_summarize":
            # TODO: parametrize the promp update, with an object or something
            self._update_summary_prompt(query_engine)

        self.engine = query_engine

        response_stream = query_engine.query(prompt)

        return response_stream

    def _update_summary_prompt(self, engine: BaseQueryEngine):
        # updating the tree summarize prompt according to business requirements

        new_summary_tmpl_str = (
            "You are a summarization service to help users navigate "
            "proprietary documentation of their companies.\n"
            "Always include the filepaths from the nodes in the context "
            "information to support the answer as well as further reading "
            "that may be mentioned in the docs. "
            f"Remove '{DATA_PATH}' from the metadatas' filepaths\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "If the query and the context don't make sense answer with "
            "'There is no good match in the docs for this prompt'\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

        engine.update_prompts(
            {"response_synthesizer:summary_template": new_summary_tmpl}
        )

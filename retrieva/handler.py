import logging
import sys

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from retrieva.data import DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class RagHandler():
    def __init__(self, data_path: str  = DATA_PATH) -> None:
        # generating the index
        documents = SimpleDirectoryReader(data_path).load_data()

        vector_index = VectorStoreIndex.from_documents(documents)
        vector_index.as_query_engine()
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=15)

        # global

        Settings.text_splitter = text_splitter

        # per-index
        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[text_splitter]
        )

    def user_prompt_streaming(self, prompt: str, similarity: int = 2):
        # set Logging to DEBUG for more detailed outputs
        query_engine = self.index.as_query_engine(streaming=True,
                                                  similarity_top_k=similarity)
        response_stream = query_engine.query(prompt)

        return response_stream

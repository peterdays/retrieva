import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from retrieva import ROOT_PATH
from retrieva.data import add_root
from retrieva.handler import RagHandler

# used in dev; in production pass the env variable to the containers
load_dotenv(os.path.join(ROOT_PATH, ".env"))

app = FastAPI(
    title="Retrieva API",
    description=("Retrieval Augmented Generation API with to ease the "
                 "documentation searching in companies")
)


rag_handler = RagHandler(
    index_name="SageMakerDocs",
    weaviate_url=os.environ["WEAVIATE_URL"],
    data_path=add_root(os.environ["DATA_FOLDER_PATH"]),
    cloud_based=os.environ["USE_CLOUD_PIPELINE"]
)

async def data_streamer(query: str):
    response_stream = rag_handler.get_response(query)
    for text in response_stream.response_gen:
        # return the texts as they arrive.
        yield text


@app.get('/')
async def root_func():
    return {"retrieva-api-version": "1.0.0.0"}

@app.get('/query')
async def query_rag(query: str):

    return StreamingResponse(data_streamer(query),
                             media_type='text/event-stream')

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from retrieva import ROOT_PATH
from retrieva.data import DATA_PATH
from retrieva.handler import RagHandler

# used in dev; in production pass the env variable to the containers
load_dotenv(os.path.join(ROOT_PATH, ".env"))

app = FastAPI()


rag_handler = RagHandler(
    index_name="SageMakerDocs",
    weaviate_url=os.environ["WEAVIATE_URL"],
    data_path=DATA_PATH
)

async def data_streamer(query: str):
    response_stream = rag_handler.user_prompt_streaming(query)
    for text in response_stream.response_gen:
        # return the texts as they arrive.
        yield text


@app.get('/')
async def main(query: str):

    return StreamingResponse(data_streamer(query), media_type='text/event-stream')

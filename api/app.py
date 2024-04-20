from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio


app = FastAPI()


async def fake_data_streamer(query: str):
    for i in range(10):
        yield f"{query}_{i}"
        await asyncio.sleep(0.5)


@app.get('/')
async def main(query: str):
    return StreamingResponse(fake_data_streamer(query), media_type='text/event-stream')
    # or, use:
    '''
    headers = {'X-Content-Type-Options': 'nosniff'}
    return StreamingResponse(fake_data_streamer(), headers=headers, media_type='text/plain')
    '''

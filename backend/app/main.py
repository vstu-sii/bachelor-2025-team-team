from langfuse import Langfuse
from fastapi import FastAPI
import os
app = FastAPI()
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


@app.middleware("http")
async def langfuse_trace(request, call_next):
    response = await call_next(request)
    langfuse.trace(
        name="api_call",
        input=request.url.path,
        output=str(response.status_code)
    )
    return response

from langfuse import Langfuse

langfuse = Langfuse(
    public_key="dev_public_key",
    secret_key="dev_secret_key",
    host="http://langfuse:3000"
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

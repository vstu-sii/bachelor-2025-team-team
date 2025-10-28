from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")

app = FastAPI(title="Local LLM Service")

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: Prompt):
    res = requests.post(f"{OLLAMA_API}/api/generate", json={
        "model": "llama3",
        "prompt": data.prompt
    })
    result = res.json()
    return {"response": result.get("response", "").strip()}

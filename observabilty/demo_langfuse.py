import os
import time
import requests
from uuid import uuid4

LANGFUSE_API_KEY = os.getenv("LANGFUSE_API_KEY")
LANGFUSE_URL = "https://api.langfuse.com/v1/events"  # псевдо-эндпоинт

def send_langfuse_event(kind: str, payload: dict):
    headers = {"Authorization": f"Bearer {LANGFUSE_API_KEY}"}
    data = {"kind": kind, "payload": payload, "timestamp": int(time.time()*1000)}
    # Также можно добавить обработку ошибок
    requests.post(LANGFUSE_URL, json=data, headers=headers, timeout=3)

def instrument_llm_call(user_id, prompt, model_name):
    trace_id = str(uuid4())
    send_langfuse_event("llm_call.start", {"trace_id": trace_id, "user_id": user_id, "prompt": prompt, "model": model_name})
    start = time.time()
    # Здесь может быть вызов LLM(пример)
    response = call_llm(prompt)
    duration_ms = int((time.time()-start)*1000)
    send_langfuse_event("llm_call.end", {"trace_id": trace_id, "response": response, "latency_ms": duration_ms})
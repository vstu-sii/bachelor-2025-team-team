from fastapi import FastAPI, HTTPException
from pathlib import Path
import uuid
import pika
import json
import subprocess
import logging
from dotenv import load_dotenv

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ml.models.baseline import HRBaseline

load_dotenv()

app = FastAPI(
    title="HR Evaluation API",
    description="API для оценки кандидатов и генерации вопросов для интервью",
    version="1.0"
)

hr_system = HRBaseline()
logging.basicConfig(level=logging.INFO)

# Запуск HR worker при старте сервера
@app.on_event("startup")
def launch_hr_worker():
    subprocess.Popen(["python", "-m", "backend.routers.worker"])

@app.post("/evaluate-candidate", tags=["HR"], summary="Оценить соответствие кандидата вакансии")
async def evaluate_candidate(
    resume_analysis: dict,
    vacancy_data: dict,
    criteria_weights: dict = None,
    candidate_id: str = None
):
    task_id = str(uuid.uuid4())
    
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
        channel = connection.channel()
        channel.queue_declare(queue="hr_tasks_queue")
        
        message = {
            "task_id": task_id,
            "task_type": "evaluation",
            "task_data": {
                "resume_analysis": resume_analysis,
                "vacancy_data": vacancy_data,
                "criteria_weights": criteria_weights,
                "candidate_id": candidate_id or f"candidate_{task_id}"
            }
        }
        
        channel.basic_publish(
            exchange="", 
            routing_key="hr_tasks_queue", 
            body=json.dumps(message)
        )
        connection.close()
        
    except Exception as e:
        logging.error(f"Ошибка очереди: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка очереди: {e}")

    return {"task_id": task_id, "status": "queued", "type": "evaluation"}

@app.post("/generate-questions", tags=["HR"], summary="Сгенерировать вопросы для интервью")
async def generate_questions(
    resume_analysis: dict,
    vacancy_requirements: dict,
    skill_gaps: list = None,
    strengths: list = None,
    experience_summary: str = None
):
    task_id = str(uuid.uuid4())
    
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
        channel = connection.channel()
        channel.queue_declare(queue="hr_tasks_queue")
        
        message = {
            "task_id": task_id,
            "task_type": "questions",
            "task_data": {
                "resume_analysis": resume_analysis,
                "vacancy_requirements": vacancy_requirements,
                "skill_gaps": skill_gaps or [],
                "strengths": strengths or [],
                "experience_summary": experience_summary or ""
            }
        }
        
        channel.basic_publish(
            exchange="", 
            routing_key="hr_tasks_queue", 
            body=json.dumps(message)
        )
        connection.close()
        
    except Exception as e:
        logging.error(f"Ошибка очереди: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка очереди: {e}")

    return {"task_id": task_id, "status": "queued", "type": "questions"}

@app.get("/task-result/{task_id}", tags=["HR"], summary="Получить результат задачи")
async def get_task_result(task_id: str):
    # Проверяем оба возможных пути результатов
    evaluation_path = Path(f"./hr_results/{task_id}_evaluation.json")
    questions_path = Path(f"./hr_results/{task_id}_questions.json")
    
    result_path = None
    task_type = None
    
    if evaluation_path.exists():
        result_path = evaluation_path
        task_type = "evaluation"
    elif questions_path.exists():
        result_path = questions_path
        task_type = "questions"
    else:
        return {"status": "processing", "task_id": task_id}

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        if "error" in result_data.get("result", {}):
            return {
                "status": "error", 
                "task_id": task_id,
                "error": result_data["result"]["error"]
            }
        
        return {
            "status": "completed",
            "task_id": task_id,
            "task_type": task_type,
            "processed_at": result_data.get("processed_at"),
            "result": result_data.get("result")
        }
        
    except Exception as e:
        logging.error(f"Ошибка чтения результата: {e}")
        return {"status": "error", "task_id": task_id, "error": str(e)}

@app.post("/full-evaluation", tags=["HR"], summary="Полная оценка кандидата (синхронно)")
async def full_evaluation_sync(
    resume_analysis: dict,
    vacancy_data: dict,
    criteria_weights: dict = None,
    candidate_id: str = None
):
    """Синхронная оценка без использования очереди"""
    try:
        result = hr_system.evaluate_candidate_full(
            resume_analysis=resume_analysis,
            vacancy_data=vacancy_data,
            criteria_weights=criteria_weights,
            candidate_id=candidate_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logging.error(f"Ошибка оценки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка оценки: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
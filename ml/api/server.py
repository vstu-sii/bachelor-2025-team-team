from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from pydantic import BaseModel
import os
import sys
import shutil
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import HRBaseline
from ml.utils.file_parser import FileParser

from dotenv import load_dotenv

# ЗАГРУЖАЕМ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ПЕРВЫМ ДЕЛОМ
load_dotenv()

app = FastAPI(
    title="HR AI Assistant API",
    description="API для анализа резюме, генерации вопросов для интервью и сопоставления с вакансиями",
    version="1.0.0"
)

class MatchRequest(BaseModel):
    analysis_result: dict
    vacancy_data: dict  # или str, если текст вакансии

class QuestionsRequest(BaseModel):
    analysis_result: dict
    vacancy_requirements: dict  # или str


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные модели
pipeline = None

# Создаем директорию для загруженных файлов
UPLOAD_DIR = "../../data/uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при запуске"""
    global pipeline
    pipeline = HRBaseline()
    print("✅ API сервер запущен с моделями:")
    print("   - Mistral для анализа резюме")

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "HR AI Assistant API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "analyze_resume": "/analyze-resume",
            "generate_interview": "/generate-interview-plan", 
            "match_vacancy": "/match-vacancy",
            "test_parser": "/test-file-parser"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    pipeline_status = "available" if pipeline else "unavailable"
    
    return {
        "status": "healthy",
        "pipeline_model": pipeline_status,
        "upload_directory": UPLOAD_DIR,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/extract-resume")
async def extract_resume(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = FileParser.extract_text_from_file(file_path)
    cleaned_text = FileParser.clean_extracted_text(resume_text)

    analysis_result = await pipeline.extract_data_from_resume(cleaned_text)
    return {"status": "success", "analysis_result": analysis_result}


@app.post("/match-vacancy")
async def match_vacancy(request: MatchRequest):
    matching_result = await pipeline.evaluate_candidate_match(
        request.analysis_result, request.vacancy_data
    )
    return {
        "status": "success",
        "resume_analysis": request.analysis_result,
        "vacancy_data": request.vacancy_data,
        "matching_result": matching_result
    }


@app.post("/generate-interview-plan")
async def generate_interview_plan(request: QuestionsRequest):
    questions_result = await pipeline.generate_interview_questions(
        resume_analysis=request.analysis_result,
        vacancy_requirements=request.vacancy_requirements
    )
    return {
        "status": "success",
        "analysis_result": request.analysis_result,
        "interview_plan": questions_result,
        "vacancy_requirements": request.vacancy_requirements
    }

@app.get("/test-file-parser")
async def test_file_parser():
    """Тестовый endpoint для проверки парсера"""
    try:
        # Создаем тестовый файл
        test_file_path = os.path.join(UPLOAD_DIR, "test_resume.txt")
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write("Тестовое резюме\nPython разработчик\nОпыт: 3 года\nНавыки: Python, Django, PostgreSQL")
        
        text = FileParser.extract_text_from_file(test_file_path)
        cleaned_text = FileParser.clean_extracted_text(text)
        
        return {
            "status": "success",
            "original_text": text,
            "cleaned_text": cleaned_text,
            "original_length": len(text),
            "cleaned_length": len(cleaned_text)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/list-uploaded-files")
async def list_uploaded_files():
    """Список загруженных файлов"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
        
        return {
            "status": "success",
            "upload_directory": UPLOAD_DIR,
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения директории: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",  # или ваш путь к файлу
        host="0.0.0.0",
        port=8000,
        reload=True
    )
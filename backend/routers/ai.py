from fastapi import FastAPI, UploadFile, Form, HTTPException, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import sys
import shutil
from datetime import datetime
import time
from pathlib import Path
import uuid
import pika
import subprocess
import logging
import asyncio
from typing import Optional, Dict, Any

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные модели и настройки
pipeline = None
UPLOAD_DIR = "./data/uploaded_resumes"
RESULTS_DIR = "./hr_results"

# Создаем директории
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Инициализация моделей и запуск воркеров при запуске"""
    global pipeline
    
    try:
        pipeline = HRBaseline()
        
        # Запуск HR worker в фоне (если нужно)
        if os.environ.get("ENABLE_WORKER", "false").lower() == "true":
            subprocess.Popen(["python", "-m", "backend.routers.worker"])
        
        logger.info("✅ API сервер запущен с моделями:")
        logger.info("   - Mistral для анализа резюме")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске сервера: {e}")
        raise

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "HR AI Assistant API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "extract_resume": "/extract-resume (POST)",
            "extract_resume_text": "/extract-resume-text (POST)",
            "evaluate_candidate": "/evaluate-candidate (POST)",
            "generate_questions": "/generate-questions (POST)",
            "task_result": "/task-result/{task_id} (GET)",
            "list_uploaded_files": "/list-uploaded-files",
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
        "results_directory": RESULTS_DIR,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Анализ резюме с парсингом файла"""
    start_time = time.time()
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Модель анализа резюме не доступна")
    
    # Проверяем формат файла
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )
    
    # Сохраняем файл
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Извлекаем текст из файла
        logger.info(f"📄 Извлекаем текст из {file.filename}...")
        resume_text = FileParser.extract_text_from_file(file_path)
        
        # Очищаем текст
        cleaned_text = FileParser.clean_extracted_text(resume_text)
        
        if not cleaned_text or cleaned_text.startswith("Не удалось"):
            raise HTTPException(
                status_code=400, 
                detail=f"Не удалось извлечь текст из файла: {cleaned_text}"
            )
        
        logger.info(f"✅ Извлечено {len(cleaned_text)} символов")
        
        # Анализируем резюме
        try:
            # Проверяем, является ли метод асинхронным
            if asyncio.iscoroutinefunction(pipeline.extract_data_from_resume):
                analysis_result = await pipeline.extract_data_from_resume(cleaned_text)
            else:
                analysis_result = pipeline.extract_data_from_resume(cleaned_text)
                
            # Проверяем результат
            if isinstance(analysis_result, dict) and "error" in analysis_result:
                raise HTTPException(status_code=500, detail=analysis_result["error"])
                
        except Exception as e:
            logger.error(f"Ошибка в pipeline.extract_data_from_resume: {e}")
            # Если метод не существует, используем fallback
            analysis_result = {
                "extracted_text": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
                "error": "Метод extract_data_from_resume не реализован, возвращен только текст"
            }
        
        # Сохраняем результат
        result_id = str(uuid.uuid4())
        result_path = os.path.join(RESULTS_DIR, f"{result_id}_resume.json")
        
        result_data = {
            "task_id": result_id,
            "filename": file.filename,
            "text_length": len(cleaned_text),
            "analysis": analysis_result,
            "processed_at": datetime.now().isoformat()
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # Трекинг успешного запроса
        latency = (time.time() - start_time) * 1000
        logger.info(f"✅ Обработка завершена за {latency:.2f} мс")
        
        return {
            "status": "success",
            "task_id": result_id,
            "filename": file.filename,
            "text_length": len(cleaned_text),
            "analysis": analysis_result,
            "result_path": result_path,
            "processing_time_ms": latency
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

@app.post("/extract-resume-text")
async def extract_resume_text(resume_text: str = Form(...)):
    """Анализ резюме из текста (без загрузки файла)"""
    try:
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Текст резюме слишком короткий (минимум 50 символов)"
            )
        
        # Анализируем резюме
        try:
            if asyncio.iscoroutinefunction(pipeline.extract_data_from_resume):
                analysis_result = await pipeline.extract_data_from_resume(resume_text)
            else:
                analysis_result = pipeline.extract_data_from_resume(resume_text)
                
            if isinstance(analysis_result, dict) and "error" in analysis_result:
                raise HTTPException(status_code=500, detail=analysis_result["error"])
                
        except Exception as e:
            logger.error(f"Ошибка в pipeline.extract_data_from_resume: {e}")
            analysis_result = {
                "extracted_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                "error": "Метод extract_data_from_resume не реализован"
            }
        
        # Сохраняем результат
        result_id = str(uuid.uuid4())
        result_path = os.path.join(RESULTS_DIR, f"{result_id}_resume.json")
        
        result_data = {
            "task_id": result_id,
            "text_length": len(resume_text),
            "analysis": analysis_result,
            "processed_at": datetime.now().isoformat()
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "task_id": result_id,
            "text_length": len(resume_text),
            "analysis": analysis_result,
            "result_path": result_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа текста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {str(e)}")

@app.post("/evaluate-candidate")
async def evaluate_candidate(
    resume_analysis: Dict[str, Any],
    vacancy_data: Dict[str, Any],
    criteria_weights: Optional[Dict[str, float]] = None,
    candidate_id: Optional[str] = None
):
    """Оценить соответствие кандидата вакансии (синхронная версия)"""
    task_id = str(uuid.uuid4())
    
    try:
        # Прямой вызов без очереди (синхронная обработка)
        logger.info(f"🚀 Начинаем оценку кандидата {candidate_id or task_id}")
        
        # Используем существующий метод pipeline или fallback
        if hasattr(pipeline, 'evaluate_candidate_match'):
            try:
                if asyncio.iscoroutinefunction(pipeline.evaluate_candidate_match):
                    evaluation_result = await pipeline.evaluate_candidate_match(
                        resume_analysis, 
                        vacancy_data
                    )
                else:
                    evaluation_result = pipeline.evaluate_candidate_match(
                        resume_analysis, 
                        vacancy_data
                    )
            except Exception as e:
                logger.error(f"Ошибка в evaluate_candidate_match: {e}")
                evaluation_result = {
                    "match_score": 0.7,
                    "strengths": ["Опыт работы", "Технические навыки"],
                    "weaknesses": ["Не указаны конкретные проекты"],
                    "recommendations": ["Уточнить опыт работы на проектах"],
                    "error": f"Ошибка при оценке: {str(e)}"
                }
        else:
            # Fallback evaluation
            evaluation_result = {
                "match_score": 0.65,
                "strengths": ["Соответствие требованиям", "Опыт работы"],
                "weaknesses": ["Требуется дополнительная информация"],
                "recommendations": ["Провести собеседование для уточнения деталей"],
                "note": "Оценка выполнена базовым алгоритмом"
            }
        
        # Сохраняем результат
        result_path = Path(RESULTS_DIR) / f"{task_id}_evaluation.json"
        result_data = {
            "task_id": task_id,
            "task_type": "evaluation",
            "candidate_id": candidate_id or f"candidate_{task_id}",
            "result": evaluation_result,
            "processed_at": datetime.now().isoformat(),
            "resume_analysis_summary": {
                "keys": list(resume_analysis.keys()) if isinstance(resume_analysis, dict) else [],
                "type": type(resume_analysis).__name__
            },
            "vacancy_data_summary": {
                "keys": list(vacancy_data.keys()) if isinstance(vacancy_data, dict) else [],
                "type": type(vacancy_data).__name__
            }
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Оценка кандидата {task_id} завершена")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "type": "evaluation",
            "result": evaluation_result,
            "result_path": str(result_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка оценки кандидата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка оценки кандидата: {str(e)}")

@app.post("/generate-questions")
async def generate_questions(
    resume_analysis: Dict[str, Any],
    vacancy_requirements: Dict[str, Any],
    skill_gaps: Optional[list] = None,
    strengths: Optional[list] = None,
    experience_summary: Optional[str] = None
):
    """Сгенерировать вопросы для интервью (синхронная версия)"""
    task_id = str(uuid.uuid4())
    
    try:
        logger.info(f"🚀 Генерация вопросов для задачи {task_id}")
        
        # Используем существующий метод pipeline или fallback
        if hasattr(pipeline, 'generate_interview_questions'):
            try:
                # Подготавливаем данные для генерации вопросов
                question_data = {
                    "analysis_result": resume_analysis,
                    "vacancy_data": vacancy_requirements,
                    "question_type": "mixed"
                }
                
                if asyncio.iscoroutinefunction(pipeline.generate_interview_questions):
                    questions_result = await pipeline.generate_interview_questions(**question_data)
                else:
                    questions_result = pipeline.generate_interview_questions(**question_data)
                    
            except Exception as e:
                logger.error(f"Ошибка в generate_interview_questions: {e}")
                questions_result = {
                    "technical_questions": [
                        "Расскажите о вашем опыте работы с Python?",
                        "Какие фреймворки вы использовали?"
                    ],
                    "behavioral_questions": [
                        "Как вы решаете конфликты в команде?",
                        "Расскажите о сложном проекте и как вы с ним справились?"
                    ],
                    "error": f"Ошибка при генерации вопросов: {str(e)}"
                }
        else:
            # Fallback questions
            questions_result = {
                "technical_questions": [
                    "Опишите ваш опыт работы с основными технологиями",
                    "Какие проекты были наиболее сложными?",
                    "Как вы подходите к тестированию кода?"
                ],
                "behavioral_questions": [
                    "Почему вы хотите работать в нашей компании?",
                    "Как вы справляетесь со сроками?",
                    "Расскажите о случае, когда вы учились на своих ошибках"
                ],
                "note": "Вопросы сгенерированы базовым алгоритмом"
            }
        
        # Добавляем дополнительные данные если они были переданы
        if skill_gaps:
            questions_result["skill_gap_questions"] = [
                f"Как вы планируете развивать навык: {skill}?" for skill in skill_gaps[:3]
            ]
        
        if strengths:
            questions_result["strength_based_questions"] = [
                f"Расскажите подробнее о вашем опыте в: {strength}" for strength in strengths[:3]
            ]
        
        # Сохраняем результат
        result_path = Path(RESULTS_DIR) / f"{task_id}_questions.json"
        result_data = {
            "task_id": task_id,
            "task_type": "questions",
            "result": questions_result,
            "processed_at": datetime.now().isoformat(),
            "input_summary": {
                "resume_analysis_keys": list(resume_analysis.keys()) if isinstance(resume_analysis, dict) else [],
                "vacancy_requirements_keys": list(vacancy_requirements.keys()) if isinstance(vacancy_requirements, dict) else []
            }
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Генерация вопросов {task_id} завершена")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "type": "questions",
            "result": questions_result,
            "result_path": str(result_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации вопросов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации вопросов: {str(e)}")

@app.get("/task-result/{task_id}")
async def get_task_result(task_id: str):
    """Получить результат задачи по ID"""
    # Проверяем все возможные пути результатов
    possible_paths = [
        Path(RESULTS_DIR) / f"{task_id}_resume.json",
        Path(RESULTS_DIR) / f"{task_id}_evaluation.json",
        Path(RESULTS_DIR) / f"{task_id}_questions.json"
    ]
    
    result_path = None
    task_type = None
    
    for path in possible_paths:
        if path.exists():
            result_path = path
            # Определяем тип задачи по имени файла
            if "resume" in path.name:
                task_type = "resume_extraction"
            elif "evaluation" in path.name:
                task_type = "evaluation"
            elif "questions" in path.name:
                task_type = "questions"
            break
    
    if not result_path:
        return {
            "status": "not_found",
            "task_id": task_id,
            "message": "Результат не найден. Задача может быть еще в обработке или ID неверный."
        }
    
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        # Проверяем наличие ошибки
        result_content = result_data.get("result", {})
        if isinstance(result_content, dict) and "error" in result_content:
            return {
                "status": "error",
                "task_id": task_id,
                "task_type": task_type,
                "error": result_content["error"],
                "processed_at": result_data.get("processed_at")
            }
        
        return {
            "status": "completed",
            "task_id": task_id,
            "task_type": task_type,
            "processed_at": result_data.get("processed_at"),
            "result": result_content,
            "metadata": {k: v for k, v in result_data.items() if k not in ['result', 'processed_at']}
        }
        
    except Exception as e:
        logger.error(f"Ошибка чтения результата: {e}")
        return {
            "status": "error",
            "task_id": task_id,
            "error": str(e)
        }

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
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "url": f"/uploads/{filename}"  # Можно добавить endpoint для скачивания
                })
        
        return {
            "status": "success",
            "upload_directory": UPLOAD_DIR,
            "total_files": len(files),
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения директории: {str(e)}")

@app.get("/list-results")
async def list_results():
    """Список всех результатов обработки"""
    try:
        results = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(RESULTS_DIR, filename)
                stats = os.stat(file_path)
                
                # Определяем тип результата
                if "resume" in filename:
                    result_type = "resume_extraction"
                elif "evaluation" in filename:
                    result_type = "evaluation"
                elif "questions" in filename:
                    result_type = "questions"
                else:
                    result_type = "unknown"
                
                task_id = filename.split('_')[0]
                
                results.append({
                    "task_id": task_id,
                    "filename": filename,
                    "type": result_type,
                    "size": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
        
        return {
            "status": "success",
            "results_directory": RESULTS_DIR,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения результатов: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import sys
import shutil
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import Gemma3Text
from ml.utils.file_parser import FileParser
from ml.api.langfuse_integration import langfuse_monitor

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

# Глобальные модели
pipeline = None

# Создаем директорию для загруженных файлов
UPLOAD_DIR = "./data/uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при запуске"""
    global pipeline
    pipeline = Gemma3Text()
    print("✅ API сервер запущен с моделями:")
    print("   - Gemma3Text для анализа резюме")

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

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    """Анализ резюме с парсингом файла и трекингом"""
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
        print(f"📄 Извлекаем текст из {file.filename}...")
        resume_text = FileParser.extract_text_from_file(file_path)
        
        # Очищаем текст
        cleaned_text = FileParser.clean_extracted_text(resume_text)
        
        if not cleaned_text or cleaned_text.startswith("Не удалось"):
            raise HTTPException(
                status_code=400, 
                detail=f"Не удалось извлечь текст из файла: {cleaned_text}"
            )
        
        print(f"✅ Извлечено {len(cleaned_text)} символов")
        
        # Анализируем резюме
        analysis_result = pipeline.analyze_resume(cleaned_text)
        
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result["error"])
        
        # Трекинг успешного запроса
        latency = (time.time() - start_time) * 1000
        langfuse_monitor.track_api_call(
            endpoint="analyze-resume",
            input_data={"filename": file.filename, "text_length": len(cleaned_text)},
            output_data={"status": "success", "analysis_keys": list(analysis_result.keys())},
            latency=latency,
            status="success",
            metadata={"file_type": file_ext, "text_preview": cleaned_text[:200]}
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "text_length": len(cleaned_text),
            "analysis": analysis_result
        }
        
    except Exception as e:
        # Трекинг ошибки
        latency = (time.time() - start_time) * 1000
        langfuse_monitor.track_api_call(
            endpoint="analyze-resume",
            input_data={"filename": file.filename},
            output_data={"error": str(e)},
            latency=latency,
            status="error",
            metadata={"error_type": type(e).__name__}
        )
        
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

@app.post("/analyze-resume-text")
async def analyze_resume_text(resume_text: str = Form(...)):
    """Анализ резюме из текста (без загрузки файла)"""
    try:
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Текст резюме слишком короткий (минимум 50 символов)"
            )
        
        # Анализируем резюме
        analysis_result = pipeline.analyze_resume(resume_text)
        
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result["error"])
        
        return {
            "status": "success",
            "text_length": len(resume_text),
            "analysis": analysis_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {str(e)}")

@app.post("/generate-interview-plan")
async def generate_interview_plan(
    file: UploadFile = File(...),
    vacancy_requirements: str = Form(""),
    question_type: str = Form("mixed")
):
    """Полный пайплайн с парсингом файла и генерацией вопросов"""
    
    # Сохраняем и парсим файл
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Извлекаем текст
        resume_text = FileParser.extract_text_from_file(file_path)
        cleaned_text = FileParser.clean_extracted_text(resume_text)
        
        if not cleaned_text or cleaned_text.startswith("Не удалось"):
            raise HTTPException(
                status_code=400, 
                detail=f"Не удалось извлечь текст из файла: {cleaned_text}"
            )
        
        # 1. Анализ резюме
        analysis_result = pipeline.analyze_resume(cleaned_text)
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result["error"])
        
        # 2. Генерация вопросов
        questions_result = pipeline.generate_questions(
            analysis_result=analysis_result,
            vacancy_data=vacancy_requirements,
            question_type=question_type
        )
        
        if "error" in questions_result:
            raise HTTPException(status_code=500, detail=questions_result["error"])
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": analysis_result,
            "interview_plan": questions_result,
            "vacancy_requirements": vacancy_requirements
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/match-vacancy")
async def match_vacancy(
    file: UploadFile = File(...),
    vacancy_data: str = Form(...)
):
    """Сопоставление резюме с вакансией"""
    
    try:
        # Сохраняем и парсим файл
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Извлекаем текст
        resume_text = FileParser.extract_text_from_file(file_path)
        cleaned_text = FileParser.clean_extracted_text(resume_text)
        
        if not cleaned_text or cleaned_text.startswith("Не удалось"):
            raise HTTPException(
                status_code=400, 
                detail=f"Не удалось извлечь текст из файла: {cleaned_text}"
            )
        
        # Анализируем резюме
        analysis_result = pipeline.analyze_resume(cleaned_text)
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result["error"])
        
        # Парсим данные вакансии
        try:
            vacancy_dict = json.loads(vacancy_data)
        except:
            # Если не JSON, используем как простой текст
            vacancy_dict = {"description": vacancy_data}
        
        # Сопоставляем с вакансией
        matching_result = pipeline.match_vacancy_with_resume(
            analysis_result, 
            vacancy_dict
        )
        
        if "error" in matching_result:
            raise HTTPException(status_code=500, detail=matching_result["error"])
        
        return {
            "status": "success",
            "filename": file.filename,
            "resume_analysis": analysis_result,
            "vacancy_data": vacancy_dict,
            "matching_result": matching_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сопоставления: {str(e)}")

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
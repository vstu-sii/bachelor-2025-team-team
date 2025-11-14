# langfuse_pipeline.py
import os
import time
import json
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import Runnable, RunnableSequence

# === Настройки окружения ===
load_dotenv()
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# === Handler для LangChain/Langfuse ===
langfuse_handler = CallbackHandler()


# === Этап 1 — Анализ резюме ===
class AnalyzeResumeRunnable(Runnable):
    def __init__(self, gemma):
        self.gemma = gemma

    def invoke(self, inputs, config=None):
        start = time.time()
        file_path = inputs.get("file_path")
        result = self.gemma.analyze_resume(file_path)
        duration = round(time.time() - start, 2)

        return {
            "stage": "analyze_resume",
            "input": {"file_path": file_path},
            "output": result,
            "duration_sec": duration
        }


# === Этап 2 — Сопоставление с вакансией ===
class MatchVacancyRunnable(Runnable):
    def __init__(self, gemma):
        self.gemma = gemma

    def invoke(self, inputs, config=None):
        start = time.time()

        # Ожидаем, что вход — это результат предыдущего шага (output) и vacancy_data
        resume_analysis = inputs.get("output")
        vacancy_data = inputs.get("vacancy_data", {})

        result = self.gemma.match_vacancy_with_resume(resume_analysis, vacancy_data)
        duration = round(time.time() - start, 2)

        return {
            "stage": "match_vacancy_with_resume",
            "input": {"resume_analysis": resume_analysis, "vacancy_data": vacancy_data},
            "output": result,
            "duration_sec": duration
        }


# === Этап 3 — Генерация вопросов ===
class GenerateQuestionsRunnable(Runnable):
    def __init__(self, gemma):
        self.gemma = gemma

    def invoke(self, inputs, config=None):
        start = time.time()

        analysis_result = inputs.get("output")
        vacancy_data = inputs.get("vacancy_data", {})

        result = self.gemma.generate_questions(analysis_result, vacancy_data)
        duration = round(time.time() - start, 2)

        return {
            "stage": "generate_questions",
            "input": {"analysis_result": analysis_result, "vacancy_data": vacancy_data},
            "output": result,
            "duration_sec": duration
        }


# === Основной пайплайн ===
def run_full_pipeline(gemma, file_path, vacancy_data):
    """
    Запускает три этапа: анализ резюме -> сопоставление -> генерация вопросов.
    Логирование — через langfuse_handler (CallbackHandler).
    """
    analyze_runnable = AnalyzeResumeRunnable(gemma).with_config(run_name="resume_analysis")
    match_runnable = MatchVacancyRunnable(gemma).with_config(run_name="vacancy_matching")
    question_runnable = GenerateQuestionsRunnable(gemma).with_config(run_name="question_generation")

    # Создаём последовательность. Подаём initial inputs: file_path и vacancy_data.
    chain = RunnableSequence(first=analyze_runnable, middle=[match_runnable], last=question_runnable)\
        .with_config(run_name="full_resume_pipeline")

    # В конфиге передаём callback handler для логирования в Langfuse
    result = chain.invoke(
        {"file_path": file_path, "vacancy_data": vacancy_data},
        config={"callbacks": [langfuse_handler]}
    )

    return result


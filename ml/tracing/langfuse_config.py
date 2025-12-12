import os
import time
import json
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from transformers import AutoTokenizer

# Загружаем ключи
load_dotenv()

# Инициализируем Langfuse клиент
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Токенайзер для подсчета токенов
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# --- Подсчёт токенов и стоимости ---
def count_tokens_and_cost(prompt: str, output: str, model: str = "mistral-medium"):
    input_tokens = len(tokenizer.encode(prompt))
    output_tokens = len(tokenizer.encode(output))
    total_tokens = input_tokens + output_tokens

    # Тарифы для Mistral (пример)
    price_per_input = 0.25 / 1_000_000
    price_per_output = 0.25 / 1_000_000
    cost = input_tokens * price_per_input + output_tokens * price_per_output

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }, cost

# --- HR функции с наблюдением ---

@observe(as_type="generation")
def evaluate_candidate(resume_analysis: dict, vacancy_data: dict, criteria_weights: dict = None):
    """Оценка кандидата с автоматической трассировкой"""
    
    # Обновляем информацию о генерации
    langfuse.update_current_generation(
        name="candidate_evaluation",
        input={
            "resume_analysis": resume_analysis,
            "vacancy_data": vacancy_data,
            "criteria_weights": criteria_weights
        },
        model="mistral-medium",
        metadata={"task_type": "candidate_evaluation"}
    )
    
    # Имитируем вызов модели (замени на реальный вызов Mistral)
    start_time = time.time()
    
    # Тут будет реальный вызов твоей HR системы
    result = {
        "overall_score": 85,
        "is_suitable": True,
        "critical_issues": [],
        "match_breakdown": {
            "job_title_match": {"score": 90},
            "experience_match": {"score": 80}
        }
    }
    
    duration = time.time() - start_time
    
    # Подсчет токенов
    prompt_text = f"Оцени кандидата: {resume_analysis} для вакансии: {vacancy_data}"
    output_text = json.dumps(result, ensure_ascii=False)
    usage_details, cost_details = count_tokens_and_cost(prompt_text, output_text)
    
    # Обновляем usage и cost
    langfuse.update_current_generation(
        output=result,
        usage_details=usage_details,
        cost_details={
            "input": cost_details,
            "output": cost_details,
            "total": cost_details * 2
        },
        metadata={"duration_sec": round(duration, 2)}
    )
    
    return result

@observe(as_type="generation") 
def generate_interview_questions(resume_analysis: dict, vacancy_requirements: dict, skill_gaps: list = None):
    """Генерация вопросов для интервью"""
    
    langfuse.update_current_generation(
        name="questions_generation",
        input={
            "resume_analysis": resume_analysis,
            "vacancy_requirements": vacancy_requirements,
            "skill_gaps": skill_gaps
        },
        model="mistral-medium",
        metadata={"task_type": "questions_generation"}
    )
    
    # Имитируем вызов модели
    start_time = time.time()
    
    result = {
        "questions": [
            {
                "type": "технический",
                "question": "Расскажите о вашем опыте с Python",
                "purpose": "Проверить технические навыки"
            },
            {
                "type": "поведенческий", 
                "question": "Как вы решали сложные задачи в прошлых проектах?",
                "purpose": "Оценить подход к решению проблем"
            }
        ]
    }
    
    duration = time.time() - start_time
    
    # Подсчет токенов
    prompt_text = f"Сгенерируй вопросы для: {resume_analysis} требования: {vacancy_requirements}"
    output_text = json.dumps(result, ensure_ascii=False)
    usage_details, cost = count_tokens_and_cost(prompt_text, output_text)
    
    langfuse.update_current_generation(
        output=result,
        usage_details=usage_details,
        cost_details={"total": cost},
        metadata={"duration_sec": round(duration, 2)}
    )
    
    return result

# --- Основные пайплайны ---

@observe()
def evaluate_candidate_pipeline(resume_analysis: dict, vacancy_data: dict, criteria_weights: dict = None):
    """Полный пайплайн оценки кандидата"""
    
    # Оценка кандидата
    evaluation_result = evaluate_candidate(resume_analysis, vacancy_data, criteria_weights)
    
    # Генерация вопросов на основе оценки
    skill_gaps = evaluation_result.get("critical_issues", [])
    questions_result = generate_interview_questions(resume_analysis, vacancy_data, skill_gaps)
    
    return {
        "evaluation": evaluation_result,
        "interview_questions": questions_result
    }

@observe()
def optimize_criteria_weights(vacancy_data: dict, company_context: dict = None):
    """Оптимизация весов критериев"""
    
    langfuse.update_current_observation(
        name="weights_optimization",
        input={
            "vacancy_data": vacancy_data,
            "company_context": company_context
        },
        metadata={"task_type": "weights_optimization"}
    )
    
    # Имитируем оптимизацию
    result = {
        "optimized_weights": {
            "job_title_weight": 25,
            "experience_weight": 30,
            "education_weight": 15,
            "skills_weight": 30
        }
    }
    
    langfuse.update_current_observation(output=result)
    return result
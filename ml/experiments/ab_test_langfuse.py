import os
import time
import json
import random
import re
import asyncio
import requests
import pandas as pd
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dotenv import load_dotenv
from langfuse import Langfuse
import sys

# Добавляем путь для импорта Langchain
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Загрузка окружения
load_dotenv()
# --- Конфигурация Mistral для проверки ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small"

# --- Ваши оригинальные промпты (сохраняем как есть) ---

from ml.prompt_templates import ( 
    UC_MATCHING_PROMPT, 
    UC_QUESTION_GENERATION_PROMPT,
    UC_ANALYSIS_PROMPT
)

# --- Альтернативные варианты для A/B тестирования ---

from ml.experiments.promts_variants import ( 
    ANALYSIS_PROMPTS, 
    MATCHING_PROMPTS,
    QUESTION_GENERATION_PROMPTS
)

# --- Утилитные функции ---

def clean_mistral_output(output: str) -> str:
    """Очистка вывода LLM"""
    if not isinstance(output, str):
        return ""
    clean = re.sub(r"^```(?:json)?", "", output.strip(), flags=re.IGNORECASE | re.MULTILINE)
    clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\n\s*-\s*\n", "\n", clean)
    clean = re.sub(r"^\s*-\s*{", "{", clean, flags=re.MULTILINE)
    clean = re.sub(r'[\x00-\x1f\x7f]', ' ', clean)
    end = clean.rfind("}")
    if end != -1:
        clean = clean[:end+1]
    return clean.strip()

def calculate_hash(data: Any) -> str:
    """Рассчитывает хэш для данных"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()[:12]

def save_input_output(
    test_id: str,
    variant: str,
    task_type: str,
    input_data: Dict,
    output_data: Dict,
    metadata: Dict,
    base_dir: str = "/ab_test_result"
):
    """Сохраняет входные и выходные данные теста"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем структуру директорий
    test_dir = Path(base_dir) / task_type / variant / test_id
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем входные данные
    input_file = test_dir / f"input_{timestamp}.json"
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_id": test_id,
            "variant": variant,
            "task_type": task_type,
            "timestamp": timestamp,
            "input_hash": calculate_hash(input_data),
            "data": input_data
        }, f, ensure_ascii=False, indent=2)
    
    # Сохраняем выходные данные
    output_file = test_dir / f"output_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_id": test_id,
            "variant": variant,
            "task_type": task_type,
            "timestamp": timestamp,
            "input_hash": calculate_hash(input_data),
            "metadata": metadata,
            "data": output_data
        }, f, ensure_ascii=False, indent=2)
    
    # Сохраняем сводку
    summary_file = test_dir / "summary.json"
    summary = {
        "test_id": test_id,
        "variant": variant,
        "task_type": task_type,
        "created_at": timestamp,
        "updated_at": timestamp,
        "input_file": str(input_file.relative_to(base_dir)),
        "output_file": str(output_file.relative_to(base_dir)),
        "input_hash": calculate_hash(input_data),
        "metadata": metadata
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return str(test_dir)

# --- Проверка качества через Mistral ---

def check_extraction_quality(
    resume_text: str,
    extracted_data: Dict,
    variant: str
) -> Dict:
    """Проверяет качество извлечения данных"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Ты — эксперт по оценке качества парсинга резюме.
    
Исходный текст резюме (первые 1500 символов):
{resume_text[:1500]}...

Извлеченные данные:
{json.dumps(extracted_data, ensure_ascii=False, indent=2)}

Использованный промпт: {variant}

Оцени качество извлечения по критериям 0-100:
1. Полнота: все ли обязательные поля заполнены (contacts, skills, experience, education)
2. Точность: соответствуют ли данные оригинальному тексту
3. Структура: правильный ли формат JSON, все ли вложенности соблюдены
4. Консистентность: нет ли противоречий в данных

Верни JSON:
{{
    "completeness_score": число,
    "accuracy_score": число,
    "structure_score": число,
    "consistency_score": число,
    "overall_score": число,
    "missing_fields": ["поле1", "поле2"],
    "incorrect_data": ["поле: что неверно"],
    "suggestions": ["предложение1", "предложение2"]
}}"""
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "Будь объективным критиком. Оценивай строго."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    
    try:
        resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return {"error": f"API error: {resp.status_code}", "success": False}
        
        content = resp.json()["choices"][0]["message"]["content"]
        cleaned = clean_mistral_output(content)
        parsed = json.loads(cleaned)
        
        return {
            "success": True,
            "completeness_score": int(parsed.get("completeness_score", 0)),
            "accuracy_score": int(parsed.get("accuracy_score", 0)),
            "structure_score": int(parsed.get("structure_score", 0)),
            "consistency_score": int(parsed.get("consistency_score", 0)),
            "overall_score": int(parsed.get("overall_score", 0)),
            "missing_fields": parsed.get("missing_fields", []),
            "incorrect_data": parsed.get("incorrect_data", []),
            "suggestions": parsed.get("suggestions", [])
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

def check_questions_quality(
    candidate_data: Dict,
    vacancy_data: Dict,
    generated_questions: List[Dict],
    variant: str
) -> Dict:
    """Проверяет качество сгенерированных вопросов"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Ты — старший рекрутер с 10+ лет опыта.
    
Данные кандидата:
{json.dumps(candidate_data, ensure_ascii=False, indent=2)}

Требования вакансии:
{json.dumps(vacancy_data, ensure_ascii=False, indent=2)}

Сгенерированные вопросы ({len(generated_questions)}):
{json.dumps(generated_questions, ensure_ascii=False, indent=2)}

Промпт генерации: {variant}

Оцени качество вопросов по критериям 0-100:
1. Релевантность: соответствуют ли вопросы профилю кандидата и вакансии
2. Разнообразие: разные типы вопросов (технические, поведенческие и т.д.)
3. Конкретность: вопросы конкретные, а не общие
4. Полезность: помогут ли вопросы принять решение о найме
5. Баланс: не слишком ли много/мало вопросов

Верни JSON:
{{
    "relevance_score": число,
    "diversity_score": число,
    "specificity_score": число,
    "usefulness_score": число,
    "balance_score": число,
    "overall_score": число,
    "strengths": ["сильные стороны"],
    "weaknesses": ["слабые стороны"],
    "recommendations": ["рекомендации по улучшению"]
}}"""
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "Оценивай как строгий эксперт по подбору."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    
    try:
        resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return {"error": f"API error: {resp.status_code}", "success": False}
        
        content = resp.json()["choices"][0]["message"]["content"]
        cleaned = clean_mistral_output(content)
        parsed = json.loads(cleaned)
        
        return {
            "success": True,
            "relevance_score": int(parsed.get("relevance_score", 0)),
            "diversity_score": int(parsed.get("diversity_score", 0)),
            "specificity_score": int(parsed.get("specificity_score", 0)),
            "usefulness_score": int(parsed.get("usefulness_score", 0)),
            "balance_score": int(parsed.get("balance_score", 0)),
            "overall_score": int(parsed.get("overall_score", 0)),
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
            "recommendations": parsed.get("recommendations", [])
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

# --- Класс для A/B тестирования ---

class HRPromptABTest:
    """A/B тестирование промптов для HR задач"""
    
    def __init__(self, model, output_dir: str = "/ab_test_results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Счетчики тестов
        self.test_counter = 0
        
    def _generate_test_id(self, task_type: str, variant: str) -> str:
        """Генерация уникального ID теста"""
        self.test_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{task_type}_{variant}_{timestamp}_{self.test_counter:04d}"
    
    def _pick_variant(self, prompts_dict: Dict) -> Tuple[str, Dict]:
        """Выбор случайного варианта промпта"""
        variants = list(prompts_dict.keys())
        weights = [0.4, 0.3, 0.3] if len(variants) == 3 else [1/len(variants)] * len(variants)
        variant = random.choices(variants, weights=weights)[0]
        return variant, prompts_dict[variant]
    
    def test_resume_analysis(
        self,
        resume_text: str,
        file_path: Optional[str] = None,
        force_variant: Optional[str] = None
    ) -> Dict:
        """Тестирование извлечения данных из резюме"""
        start_time = time.time()
        
        # Выбираем вариант промпта
        if force_variant and force_variant in ANALYSIS_PROMPTS:
            variant = force_variant
            prompt_config = ANALYSIS_PROMPTS[variant]
        else:
            variant, prompt_config = self._pick_variant(ANALYSIS_PROMPTS)
        
        test_id = self._generate_test_id("analysis", variant)
        
        try:
            # Формируем промпт
            full_prompt = f"{prompt_config['system']}\n\n{prompt_config['human']}"
            formatted_prompt = full_prompt.format(resume_text=resume_text[:10000])
            
            # Вызываем модель (адаптируйте под вашу модель)
            if hasattr(self.model, 'analyze_resume_with_prompt'):
                result = self.model.analyze_resume_with_prompt(formatted_prompt)
            elif hasattr(self.model, 'analyze_resume'):
                # Используем встроенный промпт, но логируем наш вариант
                result = self.model.analyze_resume(resume_text)
            else:
                # Фолбэк через API
                result = self._call_api_for_analysis(formatted_prompt)
            
            duration = time.time() - start_time
            
            # Проверяем качество
            quality_check = check_extraction_quality(
                resume_text[:2000],
                result if isinstance(result, dict) else {},
                variant
            )
            
            # Сохраняем входные данные
            input_data = {
                "resume_text_preview": resume_text[:1000],
                "resume_length": len(resume_text),
                "file_path": file_path,
                "prompt_variant": variant,
                "prompt_preview": formatted_prompt[:500]
            }
            
            # Сохраняем выходные данные
            output_data = {
                "extracted_data": result,
                "quality_check": quality_check,
                "processing_time": duration,
                "prompt_hash": calculate_hash(formatted_prompt)
            }
            
            # Метаданные для трекинга
            metadata = {
                "variant": variant,
                "duration_seconds": round(duration, 2),
                "quality_score": quality_check.get("overall_score", 0) if quality_check.get("success") else 0,
                "resume_hash": calculate_hash(resume_text),
                "timestamp": datetime.now().isoformat()
            }
            
            # Сохраняем все данные
            save_dir = save_input_output(
                test_id=test_id,
                variant=variant,
                task_type="resume_analysis",
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
                base_dir=self.output_dir
            )
            
            return {
                "test_id": test_id,
                "variant": variant,
                "result": result,
                "quality_check": quality_check,
                "duration": duration,
                "metadata": metadata,
                "data_saved_to": save_dir
            }
            
        except Exception as e:
            error_data = {
                "test_id": test_id,
                "variant": variant,
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Сохраняем информацию об ошибке
            error_dir = self.output_dir / "errors" / "resume_analysis" / variant / test_id
            error_dir.mkdir(parents=True, exist_ok=True)
            
            with open(error_dir / f"error_{int(time.time())}.json", 'w') as f:
                json.dump(error_data, f, indent=2)
            
            return error_data
    
    def test_matching_evaluation(
        self,
        resume_analysis: Dict,
        vacancy_data: Dict,
        weights: Dict = None,
        force_variant: Optional[str] = None
    ) -> Dict:
        """Тестирование оценки соответствия вакансии"""
        start_time = time.time()
        
        # Выбираем вариант промпта
        if force_variant and force_variant in MATCHING_PROMPTS:
            variant = force_variant
            prompt_config = MATCHING_PROMPTS[variant]
        else:
            variant, prompt_config = self._pick_variant(MATCHING_PROMPTS)
        
        test_id = self._generate_test_id("matching", variant)
        
        # Веса по умолчанию
        default_weights = {
            "job_title_weight": 20,
            "education_weight": 15,
            "experience_weight": 25,
            "schedule_weight": 5,
            "format_weight": 5,
            "additional_weight": 30
        }
        
        if weights:
            default_weights.update(weights)
        
        try:
            # Формируем промпт
            full_prompt = f"{prompt_config['system']}\n\n{prompt_config['human']}"
            formatted_prompt = full_prompt.format(
                job_title=vacancy_data.get("job_title", "Не указана"),
                education=vacancy_data.get("education", "Не указано"),
                work_experience=vacancy_data.get("work_experience", 0),
                desired_salary=vacancy_data.get("desired_salary", "Не указана"),
                work_schedule=vacancy_data.get("work_schedule", "Полный день"),
                work_format=vacancy_data.get("work_format", "Офис"),
                additional_requirements=vacancy_data.get("additional_requirements", "Нет"),
                resume_analysis=json.dumps(resume_analysis, ensure_ascii=False),
                **default_weights
            )
            
            # Вызываем модель
            if hasattr(self.model, 'evaluate_match_with_prompt'):
                result = self.model.evaluate_match_with_prompt(formatted_prompt)
            else:
                result = self._call_api_for_evaluation(formatted_prompt)
            
            duration = time.time() - start_time
            
            # Сохраняем данные
            input_data = {
                "resume_analysis_keys": list(resume_analysis.keys()),
                "vacancy_data": vacancy_data,
                "weights": default_weights,
                "prompt_variant": variant,
                "prompt_preview": formatted_prompt[:500]
            }
            
            output_data = {
                "evaluation_result": result,
                "processing_time": duration,
                "prompt_hash": calculate_hash(formatted_prompt)
            }
            
            metadata = {
                "variant": variant,
                "duration_seconds": round(duration, 2),
                "vacancy_title": vacancy_data.get("job_title", "unknown"),
                "weights_used": default_weights,
                "timestamp": datetime.now().isoformat()
            }
            
            save_dir = save_input_output(
                test_id=test_id,
                variant=variant,
                task_type="matching_evaluation",
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
                base_dir=self.output_dir
            )
            
            
            return {
                "test_id": test_id,
                "variant": variant,
                "result": result,
                "duration": duration,
                "metadata": metadata,
                "data_saved_to": save_dir
            }
            
        except Exception as e:
            error_data = {
                "test_id": test_id,
                "variant": variant,
                "error": str(e),
                "duration": time.time() - start_time
            }
            
            error_dir = self.output_dir / "errors" / "matching" / variant / test_id
            error_dir.mkdir(parents=True, exist_ok=True)
            
            with open(error_dir / f"error_{int(time.time())}.json", 'w') as f:
                json.dump(error_data, f, indent=2)
            
            return error_data
    
    def test_question_generation(
        self,
        resume_analysis: Dict,
        vacancy_requirements: Dict,
        additional_instructions: str = "",
        force_variant: Optional[str] = None
    ) -> Dict:
        """Тестирование генерации вопросов"""
        start_time = time.time()
        
        # Выбираем вариант промпта
        if force_variant and force_variant in QUESTION_GENERATION_PROMPTS:
            variant = force_variant
            prompt_config = QUESTION_GENERATION_PROMPTS[variant]
        else:
            variant, prompt_config = self._pick_variant(QUESTION_GENERATION_PROMPTS)
        
        test_id = self._generate_test_id("questions", variant)
        
        try:
            # Формируем промпт
            full_prompt = f"{prompt_config['system']}\n\n{prompt_config['human']}"
            formatted_prompt = full_prompt.format(
                job_title=vacancy_requirements.get("job_title", "Не указана"),
                education=vacancy_requirements.get("education", "Не указано"),
                work_experience=vacancy_requirements.get("work_experience", 0),
                desired_salary=vacancy_requirements.get("desired_salary", "Не указана"),
                work_schedule=vacancy_requirements.get("work_schedule", "Полный день"),
                work_format=vacancy_requirements.get("work_format", "Офис"),
                additional_requirements=vacancy_requirements.get("additional_requirements", "Нет"),
                resume_analysis=json.dumps(resume_analysis, ensure_ascii=False),
                additional_instructions=additional_instructions
            )
            
            # Вызываем модель
            if hasattr(self.model, 'generate_questions_with_prompt'):
                result = self.model.generate_questions_with_prompt(formatted_prompt)
            else:
                result = self._call_api_for_questions(formatted_prompt)
            
            duration = time.time() - start_time
            
            # Проверяем качество вопросов
            questions_list = result.get("questions", []) if isinstance(result, dict) else []
            quality_check = check_questions_quality(
                resume_analysis,
                vacancy_requirements,
                questions_list,
                variant
            )
            
            # Сохраняем данные
            input_data = {
                "resume_analysis_keys": list(resume_analysis.keys()),
                "vacancy_requirements": vacancy_requirements,
                "additional_instructions": additional_instructions,
                "prompt_variant": variant,
                "prompt_preview": formatted_prompt[:500]
            }
            
            output_data = {
                "generated_questions": result,
                "questions_count": len(questions_list),
                "quality_check": quality_check,
                "processing_time": duration,
                "prompt_hash": calculate_hash(formatted_prompt)
            }
            
            metadata = {
                "variant": variant,
                "duration_seconds": round(duration, 2),
                "questions_count": len(questions_list),
                "quality_score": quality_check.get("overall_score", 0) if quality_check.get("success") else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            save_dir = save_input_output(
                test_id=test_id,
                variant=variant,
                task_type="question_generation",
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
                base_dir=self.output_dir
            )
            
            return {
                "test_id": test_id,
                "variant": variant,
                "result": result,
                "quality_check": quality_check,
                "duration": duration,
                "metadata": metadata,
                "data_saved_to": save_dir
            }
            
        except Exception as e:
            error_data = {
                "test_id": test_id,
                "variant": variant,
                "error": f"KeyError в форматировании промпта: {e}",
                "duration": time.time() - start_time
            }
            
            error_dir = self.output_dir / "errors" / "questions" / variant / test_id
            error_dir.mkdir(parents=True, exist_ok=True)
            
            with open(error_dir / f"error_{int(time.time())}.json", 'w') as f:
                json.dump(error_data, f, indent=2)
            
            return error_data
    
    def _call_api_for_analysis(self, prompt: str) -> Dict:
        """Фолбэк вызов API для анализа"""
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": "Ты HR-специалист. Возвращай только валидный JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }
        
        try:
            resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(clean_mistral_output(content))
        except:
            pass
        
        return {"error": "API call failed"}
    
    def _call_api_for_evaluation(self, prompt: str) -> Dict:
        """Фолбэк вызов API для оценки"""
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": "Ты эксперт по оценке кандидатов. Возвращай JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }
        
        try:
            resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(clean_mistral_output(content))
        except:
            pass
        
        return {"error": "API call failed"}
    
    def _call_api_for_questions(self, prompt: str) -> Dict:
        """Фолбэк вызов API для вопросов"""
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": "Ты рекрутер. Возвращай JSON с вопросами."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }
        
        try:
            resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(clean_mistral_output(content))
        except:
            pass
        
        return {"error": "API call failed"}

# --- Массовое тестирование ---

async def run_batch_ab_tests(
    test_cases: List[Dict],
    model,
    output_dir: str = "/ab_test_results"
) -> pd.DataFrame:
    """Запуск массового A/B тестирования"""
    
    tester = HRPromptABTest(model, output_dir)
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Тест {i}/{len(test_cases)}: {test_case.get('name', f'Test {i}')}")
        
        try:
            # Анализ резюме
            if "resume_text" in test_case:
                print("  📄 Анализ резюме...")
                result = tester.test_resume_analysis(
                    resume_text=test_case["resume_text"],
                    file_path=test_case.get("file_path")
                )
                
                if "error" not in result:
                    all_results.append({
                        "test_id": result["test_id"],
                        "task": "resume_analysis",
                        "variant": result["variant"],
                        "quality_score": result.get("quality_check", {}).get("overall_score", 0),
                        "duration": result["duration"],
                        "success": True
                    })
                    print(f"    ✅ {result['variant']}, Оценка: {result.get('quality_check', {}).get('overall_score', 0)}")
                else:
                    print(f"    ❌ Ошибка: {result['error']}")
            
            # Генерация вопросов (если есть данные кандидата и вакансии)
            if "resume_analysis" in test_case and "vacancy_requirements" in test_case:
                print("  ❓ Генерация вопросов...")
                result = tester.test_question_generation(
                    resume_analysis=test_case["resume_analysis"],
                    vacancy_requirements=test_case["vacancy_requirements"],
                    additional_instructions=test_case.get("additional_instructions", "")
                )
                
                if "error" not in result:
                    all_results.append({
                        "test_id": result["test_id"],
                        "task": "question_generation",
                        "variant": result["variant"],
                        "quality_score": result.get("quality_check", {}).get("overall_score", 0),
                        "questions_count": result.get("result", {}).get("questions_count", 0),
                        "duration": result["duration"],
                        "success": True
                    })
                    print(f"    ✅ {result['variant']}, Вопросов: {result.get('result', {}).get('questions_count', 0)}")
            
            # Оценка соответствия
            if "resume_analysis" in test_case and "vacancy_data" in test_case:
                print("  📊 Оценка соответствия...")
                result = tester.test_matching_evaluation(
                    resume_analysis=test_case["resume_analysis"],
                    vacancy_data=test_case["vacancy_data"],
                    weights=test_case.get("weights")
                )
                
                if "error" not in result:
                    all_results.append({
                        "test_id": result["test_id"],
                        "task": "matching_evaluation",
                        "variant": result["variant"],
                        "duration": result["duration"],
                        "success": True
                    })
                    print(f"    ✅ {result['variant']}, Время: {result['duration']:.2f}с")
        
        except Exception as e:
            print(f"  ❌ Критическая ошибка: {e}")
            all_results.append({
                "test_id": f"error_{i}",
                "task": "unknown",
                "variant": "error",
                "error": str(e),
                "success": False
            })
    
    # Анализ результатов
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Сохраняем сводку
        summary_file = Path(output_dir) / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print(f"\n📊 Сводка сохранена в: {summary_file}")
        
        # Анализ по вариантам
        successful = df[df["success"] == True]
        
        if not successful.empty:
            print("\n📈 АНАЛИЗ РЕЗУЛЬТАТОВ:")
            print("=" * 60)
            
            for task in successful["task"].unique():
                task_data = successful[successful["task"] == task]
                print(f"\n🔹 Задача: {task}")
                print(f"   Всего тестов: {len(task_data)}")
                
                for variant in task_data["variant"].unique():
                    variant_data = task_data[task_data["variant"] == variant]
                    
                    if task == "resume_analysis":
                        avg_score = variant_data["quality_score"].mean()
                        print(f"   - {variant}: {len(variant_data)} тестов, средняя оценка: {avg_score:.1f}")
                    
                    elif task == "question_generation":
                        avg_score = variant_data["quality_score"].mean()
                        avg_questions = variant_data["questions_count"].mean()
                        print(f"   - {variant}: {len(variant_data)} тестов, оценка: {avg_score:.1f}, вопросов: {avg_questions:.1f}")
                    
                    else:
                        avg_duration = variant_data["duration"].mean()
                        print(f"   - {variant}: {len(variant_data)} тестов, время: {avg_duration:.2f}с")
        
        return df
    
    return pd.DataFrame()


# --- Пример использования ---

def main():
    """Пример запуска A/B тестирования"""
    
    # Импортируем модель (замените на свою)
    try:
        from ml.models.baseline import HRBaseline
        model = HRBaseline()
        print("✅ Модель загружена")
    except ImportError:
        print("⚠️ Модель не найдена, используем API фолбэк")
        model = None
    
    # Создаем тестер
    tester = HRPromptABTest(model)
    
    # Пример 1: Тест анализа резюме
    print("\n🧪 ТЕСТ 1: Анализ резюме")
    
    test_resume = """Иванов Иван Иванович
    Python разработчик
    
    Контакты:
    Email: ivan@example.com
    Телефон: +7-999-123-45-67
    Город: Москва
    
    Навыки:
    - Python, Django, FastAPI
    - PostgreSQL, Redis
    - Docker, Git
    - Английский (Intermediate)
    
    Опыт работы:
    Senior Python Developer, TechCorp (2021-2024)
    - Разработка микросервисов на FastAPI
    - Оптимизация производительности
    
    Middle Python Developer, Startup Inc (2019-2021)
    - Разработка backend на Django
    - Интеграция с внешними API
    
    Образование:
    МГТУ им. Баумана, Факультет информатики
    Специалист, 2019 год
    """
    
    candidate_data = tester.test_resume_analysis(test_resume)
    print(f"Вариант: {candidate_data['variant']}")
    print(f"Оценка качества: {candidate_data.get('quality_check', {}).get('overall_score', 'N/A')}")
    print(f"Данные сохранены в: {candidate_data.get('data_saved_to', 'N/A')}")
    

    # Пример 2: Тест оценки соответствия
    print("\n🧪 ТЕСТ 2: Оценка соответствия вакансии")

    vacancy_data = {
        "job_title": "Senior Python Developer",
        "education": "Высшее техническое",
        "work_experience": 3,
        "desired_salary": "200000 руб.",
        "work_schedule": "Полный день", 
        "work_format": "Удаленно/гибрид",
        "additional_requirements": "Опыт работы с микросервисами, знание Docker"
    }

    result = tester.test_matching_evaluation(
        resume_analysis=candidate_data.get('result', {}),  # Из теста 2
        vacancy_data=vacancy_data,
        weights={
            "job_title_weight": 20,
            "education_weight": 15,
            "experience_weight": 25,
            "schedule_weight": 5,
            "format_weight": 5,
            "additional_weight": 30
        }
    )

    print(f"Вариант: {result['variant']}")
    print(f"Данные сохранены в: {result.get('data_saved_to', 'N/A')}")
    if 'result' in result:
        eval_result = result['result']
        if isinstance(eval_result, dict) and 'overall_score' in eval_result:
            print(f"Оценка соответствия: {eval_result['overall_score']}")

    # Пример 3: Тест генерации вопросов
    print("\n🧪 ТЕСТ 3: Генерация вопросов")
    
    result = tester.test_question_generation(candidate_data.get('result', {}), vacancy_data)
    print(f"Вариант: {result['variant']}")
    print(f"Сгенерировано вопросов: {result.get('result', {}).get('questions_count', 0)}")
    print(f"Оценка качества: {result.get('quality_check', {}).get('overall_score', 'N/A')}")
    
    return tester

if __name__ == "__main__":
    tester = main()
    
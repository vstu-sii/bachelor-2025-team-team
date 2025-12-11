import os
import json
import time
import re
import requests
import glob
import asyncio
import random
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import HRBaseline, langfuse

# Загружаем ключи
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-large-latest"

# Инициализируем модели
hr_system = HRBaseline()

def clean_mistral_output(output: str) -> str:
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

def post_with_retries(url, headers, payload, timeout=60, max_retries=5, base_delay=0.8):
    """Ретраи с экспоненциальной задержкой и джиттером для 429/5xx ошибок."""
    last_resp = None
    for attempt in range(1, max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        last_resp = resp

        if resp.status_code == 200:
            return resp

        retry_after = resp.headers.get("Retry-After")
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                break
            if retry_after:
                try:
                    delay = float(retry_after)
                except Exception:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            else:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(delay)
            continue
        break
    return last_resp
    
def _find_test_files(test_folders):

    """Ищет тестовые файлы в различных папках"""
    test_files = []
    for folder in test_folders:
        if os.path.exists(folder):
            # Ищем все поддерживаемые форматы
            patterns = [f"{folder}/*.pdf", f"{folder}/*.docx", f"{folder}/*.doc", f"{folder}/*.txt"]
            for pattern in patterns:
                found_files = glob.glob(pattern)
                test_files.extend(found_files)
    
    return test_files

def check_extraction_with_mistral(
   resume_files,  #Исходные файлы с резюме
   extracted_data #Извлеченные из резюме данные
):
    """
    Проверяет:
    
    """
    headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
    
    prompt = (
        "Ты проверяющий ассистент. Верни строго валидный JSON без Markdown.\n\n"
        "Задача: проверить извлеченные из резюме данные на соответсвие исходным данным и на присутствие необходимых полей.\n"
        "Необходимые поля:\n"
        " - contacts\n"
        " - skills\n"
        " - experience\n"
        " - education\n\n"
        f"Исходный файл резюме:\n{resume_files}\n\n"
        f"Извлеченные данные из резюме: {json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n"
        "Проверь:\n"
        "1) Присутствуют ли все необходивые поля.\n"
        "2) Все ли навыки были извлечены.\n"
        "3) Все ли места работы извлеклись\n"
        "4) Общую оценку извлечения данных. На сколько извлеченные данные соответствуют оригинальному резюме (из 100).\n\n"
        "Ответь ТОЛЬКО JSON следующей формы:\n"
        "{\n"
        '  "all_fields_present": true,\n'
        '  "all_skills_present": true,\n'
        '  "all_experience_companies": true,\n'
        '  "match_analysis": 100\n'
        "}\n"
    )

    payload = {
            "model": MISTRAL_MODEL,
            "messages": [
            {"role": "system", "content": "Ты проверяющий ассистент. Возвращай только JSON."},
            {"role": "user", "content": prompt}
        ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
    
    try:
        resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return {"error": f"Mistral API error: {resp.status_code}"}
        
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        
        if not content:
            return {"error": "Empty response"}
        
        cleaned = clean_mistral_output(content)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            return {"error": f"Invalid JSON: {e}", "raw_output": content}
            
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    usage = data.get("usage", {}) or {}
    return {
        "all_fields_present": bool(parsed.get("all_fields_present", False)),
        "all_skills_present": bool(parsed.get("all_skills_present", False)),
        "all_experience_companies": bool(parsed.get("all_experience_companies", False)),
        "match_analysis": int(parsed.get("match_analysis", 0)),
        "usage": usage
    }

def check_evaluation_with_mistral(
   resume_data,               #Файлы с данными из резюме
   vacancy,                   #Вакансия
   resume__with_evaluation,   #Файлы с оценками резюме
):
    """
    Проверяет:
    
    """
    headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
    
    prompt = (
        "Ты проверяющий ассистент. Верни строго валидный JSON без Markdown.\n\n"
        "Задача: проверить оценки резюме по соответствию вакансии и веса критериев оценки на адекватность и объективность. \n"
        "В полях будут оценки того, на сколько по этому критерию резюме кандидата соответствует вакансии.\n"
        "Веса критериев в сумме должны давать 1.\n"
        f"Файл с данными из резюме:\n{json.dumps(resume_data, ensure_ascii=False, indent=2)}\n\n"
        f"Файл с вакансией: {json.dumps(vacancy, ensure_ascii=False, indent=2)}\n"
        f"Файл с оценками резюме:\n{json.dumps(resume__with_evaluation, ensure_ascii=False, indent=2)}\n\n"
        "Проверь:\n"
        "1) На сколько оценка в поле job_title_match точна (от 0 до 100).\n"
        "2) Какой вес в поле job_title_match лучше всего подходит для данного критерия (от 0 до 1).\n"
        "3) На сколько оценка в поле education_match точна (от 0 до 100).\n"
        "4) Какой вес в поле education_match лучше всего подходит для данного критерия (от 0 до 1).\n"
        "5) На сколько оценка в поле experience_match точна (от 0 до 100).\n"
        "6) Какой вес в поле experience_match лучше всего подходит для данного критерия (от 0 до 1).\n"
        "7) На сколько оценка в поле schedule_match точна (от 0 до 100).\n"
        "8) Какой вес в поле schedule_match лучше всего подходит для данного критерия (от 0 до 1).\n"
        "9) На сколько оценка в поле format_match точна (от 0 до 100).\n"
        "10) Какой вес в поле format_match лучше всего подходит для данного критерия (от 0 до 1).\n"
        "11) На сколько оценка в поле additional_match точна (от 0 до 100).\n"
        "12) Какой вес в поле additional_match лучше всего подходит для данного критерия (от 0 до 1).\n"
        "13) На сколько общая оценка соответствия резюме вакансии (overall_score) соответствует действительности (от 0 до 100).\n"
        "14) На сколько поле recommendation соответствует резюме относительно вакансии (от 0 до 100).\n"
        "Ответь ТОЛЬКО JSON следующей формы:\n"
        "{\n"
        '  "job_title_match_eval": 100,\n'
        '  "job_title_match_weight": 0.5,\n'
        '  "education_match_eval": 100,\n'
        '  "education_match_weight": 0.5,\n'
        '  "experience_match_eval": 100,\n'
        '  "experience_match_weight": 0.5,\n'
        '  "schedule_match_eval": 100,\n'
        '  "schedule_match_weight": 0.5,\n'
        '  "format_match_eval": 100,\n'
        '  "format_match_weight": 0.5,\n'
        '  "additional_match_eval": 100,\n'
        '  "additional_match_weight": 0.5,\n'
        '  "overall_score_estimation": 100,\n'
        '  "recomendation_estimation": 100\n'
        "}\n"
    )

    payload = {
            "model": MISTRAL_MODEL,
            "messages": [
            {"role": "system", "content": "Ты проверяющий ассистент. Возвращай только JSON."},
            {"role": "user", "content": prompt}
        ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
    
    try:
        resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return {"error": f"Mistral API error: {resp.status_code}"}
        
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        
        if not content:
            return {"error": "Empty response"}
        
        cleaned = clean_mistral_output(content)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            return {"error": f"Invalid JSON: {e}", "raw_output": content}
            
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    usage = data.get("usage", {}) or {}
    return {
        "job_title_match_eval": int(parsed.get("job_title_match_eval", 0)),
        "job_title_match_weight": float(parsed.get("job_title_match_weight", 0)),
        "education_match_eval": int(parsed.get("education_match_eval", 0)),
        "education_match_weight": float(parsed.get("education_match_weight", 0)),
        "experience_match_eval": int(parsed.get("experience_match_eval", 0)),
        "experience_match_weight": float(parsed.get("experience_match_weight", 0)),
        "schedule_match_eval": int(parsed.get("schedule_match_eval", 0)),
        "schedule_match_weight": float(parsed.get("schedule_match_weight", 0)),
        "format_match_eval": int(parsed.get("format_match_eval", 0)),
        "format_match_weight": float(parsed.get("format_match_weight", 0)),
        "additional_match_eval": int(parsed.get("additional_match_eval", 0)),
        "additional_match_weight": float(parsed.get("additional_match_weight", 0)),
        "overall_score_estimation": int(parsed.get("overall_score_estimation", 0)),
        "recomendation_estimation": int(parsed.get("recomendation_estimation", 0)),
        "usage": usage
    }

def check_questions_with_mistral(
    extracted_data,               # Файлы с данными из резюме
    vacancy,                      # Вакансия
    questions_for_resume,         # Файлы с вопросами
):
    """
    Проверяет сгенерированные вопросы на:
    1. Релевантность данным резюме
    2. Релевантность требованиям вакансии
    3. Качество вопросов (конкретность, полезность для оценки)
    4. Разнообразие вопросов (не повторяются)
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Проверяем структуру вопросов
    if isinstance(questions_for_resume, list):
        questions_list = questions_for_resume
    elif isinstance(questions_for_resume, dict):
        if "questions" in questions_for_resume:
            questions_list = questions_for_resume["questions"]
        elif "generated_questions" in questions_for_resume:
            questions_list = questions_for_resume["generated_questions"]
        else:
            questions_list = list(questions_for_resume.values())
    else:
        return {"error": "Неверный формат вопросов"}
    
    prompt = (
        "Ты проверяющий ассистент для HR-системы. Верни строго валидный JSON без Markdown.\n\n"
        "Задача: проверить качество сгенерированных вопросов для кандидата на основе его резюме и вакансии.\n\n"
        "КРИТЕРИИ ПРОВЕРКИ:\n"
        "1. Релевантность резюме: Вопросы должны быть основаны на данных из резюме\n"
        "2. Релевантность вакансии: Вопросы должны помогать оценить соответствие требованиям вакансии\n"
        "3. Конкретность: Вопросы должны быть конкретными, а не общими\n"
        "4. Разнообразие: Вопросы не должны повторяться и должны охватывать разные аспекты\n"
        "5. Практическая полезность: Вопросы должны помогать принять решение о найме\n\n"
        "ДАННЫЕ РЕЗЮМЕ:\n"
        f"{extracted_data}\n\n"
        "ДАННЫЕ ВАКАНСИИ:\n"
        f"{json.dumps(vacancy, ensure_ascii=False, indent=2)}\n\n"
        "СГЕНЕРИРОВАННЫЕ ВОПРОСЫ:\n"
        f"{questions_list}\n\n"
        "ПРОВЕРЬ:\n"
        "1) Релевантность_резюме: Насколько вопросы соответствуют данным из резюме (от 0 до 100)\n"
        "2) Релевантность_вакансии: Насколько вопросы помогают оценить соответствие требованиям вакансии (от 0 до 100)\n"
        "3) Конкретность_вопросов: Насколько вопросы конкретны и сфокусированы (от 0 до 100)\n"
        "4) Разнообразие_вопросов: Насколько вопросы разнообразны и покрывают разные темы (от 0 до 100)\n"
        "5) Практическая_полезность: Насколько вопросы полезны для принятия решения о найме (от 0 до 100)\n"
        "6) Количество_повторений: Сколько раз повторяются похожие вопросы (число)\n"
        "7) Рекомендации: Список рекомендаций по улучшению вопросов\n"
        "8) Лучшие_вопросы: Список 3-х лучших вопросов из предоставленных\n"
        "9) Вопросы_на_улучшение: Список 3-х вопросов, которые нужно улучшить или заменить\n\n"
        "Ответь ТОЛЬКО JSON следующей формы:\n"
        "{\n"
        '  "relevance_to_resume": 85,\n'
        '  "relevance_to_vacancy": 90,\n'
        '  "question_specificity": 75,\n'
        '  "question_diversity": 80,\n'
        '  "practical_usefulness": 88,\n'
        '  "repetition_count": 2,\n'
        '  "recommendations": ["рекомендация 1", "рекомендация 2"],\n'
        '  "best_questions": ["лучший вопрос 1", "лучший вопрос 2", "лучший вопрос 3"],\n'
        '  "questions_to_improve": ["вопрос на улучшение 1", "вопрос на улучшение 2", "вопрос на улучшение 3"]\n'
        "}\n"
    )

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "Ты опытный HR-эксперт, который оценивает качество собеседования. Возвращай только JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    
    try:
        resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code != 200:
            return {"error": f"Mistral API error: {resp.status_code}"}
        
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        
        if not content:
            return {"error": "Empty response"}
        
        cleaned = clean_mistral_output(content)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            return {"error": f"Invalid JSON: {e}", "raw_output": content}
            
    except Exception as e:
        return {"error": f"API call failed: {e}"}

    usage = data.get("usage", {}) or {}
    
    return {
        "relevance_to_resume": int(parsed.get("relevance_to_resume", 0)),
        "relevance_to_vacancy": int(parsed.get("relevance_to_vacancy", 0)),
        "question_specificity": int(parsed.get("question_specificity", 0)),
        "question_diversity": int(parsed.get("question_diversity", 0)),
        "practical_usefulness": int(parsed.get("practical_usefulness", 0)),
        "repetition_count": int(parsed.get("repetition_count", 0)),
        "recommendations": parsed.get("recommendations", []),
        "best_questions": parsed.get("best_questions", []),
        "questions_to_improve": parsed.get("questions_to_improve", []),
        "usage": usage,
        "total_questions_analyzed": len(questions_list)
    }

async def run_test_extraction():

    passed_all = passed_fields = passed_skills = passed_experience = passed_experience_name = passed_experience_description = passed_match_analysis = 0
    all_results = []

    test_files = _find_test_files([
        "data/test_resumes"
    ])
    
    total = len(test_files)
    total_match = 0
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
        
        start = time.time()
        # Извелкаем данные из резюме
        extracted_data = await hr_system.extract_data_from_resume(file_path)

        gen_latency = round(time.time() - start, 3)

        check_extraction = check_extraction_with_mistral(file_path, extracted_data)

        all_fields_present = "OK" if check_extraction.get("all_fields_present", False) else "FAIL"
        all_skills_present = "OK" if check_extraction.get("all_skills_present", False) else "FAIL"
        all_experience_companies = "OK" if check_extraction.get("all_experience_companies", False) else "FAIL"
        match_analysis = check_extraction.get("match_analysis")
        usage = check_extraction.get("usage", {}) or {}

        if all_fields_present == "OK": passed_fields += 1
        if all_skills_present == "OK": passed_skills += 1
        if all_experience_companies == "OK": passed_experience += 1
        if match_analysis >= 70 : 
            passed_match_analysis += 1
            total_match += match_analysis
        if all(v == "OK" for v in [all_fields_present, all_skills_present, all_experience_companies]) and match_analysis >= 70:
            passed_all += 1

        print(f"  Все поля: {all_fields_present}, Все навыки: {all_skills_present}, Места работы: {all_experience_companies}, Обшая оценка извлечения: {match_analysis}")
 
        all_results.append({
                "id": i,
                "all_fields_present": all_fields_present,
                "all_skills_present": all_skills_present,
                "all_experience_companies": all_experience_companies,
                "match_analysis": match_analysis,
                "gen_latency": gen_latency,
                "total_tokens": usage.get("total_tokens"),
                "error": None
            })

    # Закрываем клиент
    await hr_system.close_client()

    print(f"\n📊 Итог по {total} тестам:")
    print(f"✅ Все ли поля на месте: {passed_fields}/{total}")
    print(f"✅ Все ли навыки извлечены: {passed_skills}/{total}")
    print(f"✅ Весь ли опыт извлечен: {all_experience_companies}/{total}")
    print(f"✅ Оценки извлечения: {passed_match_analysis}/{total}")
    print(f"✅ Средняя оценка извлечения: {total_match / total}")
    print(f"✅ Все условия: {passed_all}/{total}")

async def run_test_evaluation():

    all_results = []

    test_files = _find_test_files([
        "data/test_resumes"
    ])

    vacancy = "data/vacancy.json"

    # Загружаем данные вакансии (один раз)
    try:
        with open(vacancy, 'r', encoding='utf-8') as f:
            vacancy = json.load(f)
        print(f"✅ Загружены данные вакансии: {vacancy.get('position', 'Не указана')}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных вакансии: {e}")
        return
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
    
        start = time.time()

        # Извелкаем данные из резюме
        extracted_data = await hr_system.extract_data_from_resume(file_path)
        
        #Оцениваем резюме
        eval_resumes = await hr_system.evaluate_candidate_match(extracted_data, vacancy)

        gen_latency = round(time.time() - start, 3)

        #Оцениваем оценки резюме
        check_evaluation = check_evaluation_with_mistral(file_path, vacancy, eval_resumes)

        job_title_match_eval = check_evaluation.get("job_title_match_eval") 
        job_title_match_weight = check_evaluation.get("job_title_match_weight")
        education_match_eval = check_evaluation.get("education_match_eval") 
        education_match_weight = check_evaluation.get("education_match_weight")
        experience_match_eval = check_evaluation.get("experience_match_eval") 
        experience_match_weight = check_evaluation.get("experience_match_weight")
        schedule_match_eval = check_evaluation.get("schedule_match_eval") 
        schedule_match_weight = check_evaluation.get("schedule_match_weight")
        format_match_eval = check_evaluation.get("format_match_eval") 
        format_match_weight = check_evaluation.get("format_match_weight")
        additional_match_eval = check_evaluation.get("additional_match_eval") 
        additional_match_weight = check_evaluation.get("additional_match_weight")
        overall_score_estimation = check_evaluation.get("overall_score_estimation")
        recomendation_estimation = check_evaluation.get("recomendation_estimation")
        usage = check_evaluation.get("usage", {}) or {}

        print(f"Соответствие должности: {job_title_match_eval}\n"
                f"Вес криетрия должности:{job_title_match_weight}\n"
                f"Соответствие уровня образования: {education_match_eval}\n"
                f"Вес криетрия уровня образования: {education_match_weight}\n"
                f"Соответствие опыту: {experience_match_eval}\n"
                f"Вес криетрия опыта: {experience_match_weight}\n"
                f"Соответствие графику работы: {schedule_match_eval}\n"
                f"Вес криетрия графика работы: {schedule_match_weight}\n"
                f"Соответствие формату работы: {format_match_eval}\n"
                f"Вес криетрия формата работы: {format_match_weight}\n"
                f"Соответствие дополнительным критериям: {additional_match_eval}\n"
                f"Вес криетрия дополнительных критерией: {additional_match_weight}\n"
                f"Общая оценка: {overall_score_estimation}\n"
                f"Соотвествие рекомедаций: {recomendation_estimation}\n"
               )
 
        all_results.append({
                "id": i,
                "job_title_match_eval": job_title_match_eval,
                "job_title_match_weight": job_title_match_weight,
                "education_match_eval": education_match_eval,
                "education_match_weight": education_match_weight,
                "experience_match_eval": experience_match_eval,
                "experience_match_weight": experience_match_weight,
                "schedule_match_eval": schedule_match_eval,
                "schedule_match_weight": schedule_match_weight,
                "format_match_eval": format_match_eval,
                "format_match_weight": format_match_weight,
                "additional_match_eval": additional_match_eval,
                "additional_match_weight": additional_match_weight,
                "overall_score_estimation": overall_score_estimation,
                "recomendation_estimation": recomendation_estimation,
                "gen_latency": gen_latency,
                "total_tokens": usage.get("total_tokens"),
                "error": None
            })
        
    # Закрываем клиент
    await hr_system.close_client()

async def run_test_question():
    """Тестирование генерации вопросов к резюме"""
    
    # Инициализация системы
    hr_system = HRBaseline()
    
    test_files = _find_test_files([
        "data/test_resumes"
    ])

    vacancy = "data/vacancy.json"
    
    # Загружаем данные вакансии (один раз)
    try:
        with open(vacancy, 'r', encoding='utf-8') as f:
            vacancy = json.load(f)
        print(f"✅ Загружены данные вакансии: {vacancy.get('position', 'Не указана')}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных вакансии: {e}")
        return
    
    passed_all = passed_relevance_resume = passed_relevance_vacancy = passed_specificity = passed_diversity = passed_usefulness = 0
    all_results = []
    
    total = len(test_files)
    total_relevance_resume = 0
    total_relevance_vacancy = 0
    total_specificity = 0
    total_diversity = 0
    total_usefulness = 0
    
    for i, file_path in enumerate(test_files, 1):
            
        print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
        
        start = time.time()

        # Загружаем данные
        try:
           # Извелкаем данные из резюме
            extracted_data = await hr_system.extract_data_from_resume(file_path)
            
            #Оцениваем резюме
            question_resumes = await hr_system.generate_interview_questions(extracted_data, vacancy)

            gen_latency = round(time.time() - start, 3)

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            continue
        
        # Проверяем качество вопросов
        check_result = check_questions_with_mistral(
            extracted_data=extracted_data,
            vacancy=vacancy,
            questions_for_resume=question_resumes
        )
        
        if "error" in check_result:
            print(f"❌ Ошибка проверки: {check_result['error']}")
            all_results.append({
                "id": i,
                "resume_file": os.path.basename(file_path),
                "relevance_to_resume": "FAIL",
                "relevance_to_vacancy": "FAIL",
                "question_specificity": "FAIL",
                "question_diversity": "FAIL",
                "practical_usefulness": "FAIL",
                "repetition_count": 0,
                "gen_latency": gen_latency,
                "total_tokens": 0,
                "error": check_result.get("error"),
                "total_questions": 0
            })
            continue
        
        # Извлекаем метрики
        relevance_resume_score = check_result.get("relevance_to_resume", 0)
        relevance_vacancy_score = check_result.get("relevance_to_vacancy", 0)
        specificity_score = check_result.get("question_specificity", 0)
        diversity_score = check_result.get("question_diversity", 0)
        usefulness_score = check_result.get("practical_usefulness", 0)
        repetition_count = check_result.get("repetition_count", 0)
        usage = check_result.get("usage", {}) or {}
        
        # Определяем статусы OK/FAIL (порог 70%)
        relevance_resume_ok = "OK" if relevance_resume_score >= 70 else "FAIL"
        relevance_vacancy_ok = "OK" if relevance_vacancy_score >= 70 else "FAIL"
        specificity_ok = "OK" if specificity_score >= 70 else "FAIL"
        diversity_ok = "OK" if diversity_score >= 70 else "FAIL"
        usefulness_ok = "OK" if usefulness_score >= 70 else "FAIL"
        
        # Считаем статистику
        if relevance_resume_ok == "OK": 
            passed_relevance_resume += 1
            total_relevance_resume += relevance_resume_score
        
        if relevance_vacancy_ok == "OK": 
            passed_relevance_vacancy += 1
            total_relevance_vacancy += relevance_vacancy_score
        
        if specificity_ok == "OK": 
            passed_specificity += 1
            total_specificity += specificity_score
        
        if diversity_ok == "OK": 
            passed_diversity += 1
            total_diversity += diversity_score
        
        if usefulness_ok == "OK": 
            passed_usefulness += 1
            total_usefulness += usefulness_score
        
        # Проверяем все условия
        all_conditions_ok = all(v == "OK" for v in [
            relevance_resume_ok, relevance_vacancy_ok, 
            specificity_ok, diversity_ok, usefulness_ok
        ])
        
        if all_conditions_ok:
            passed_all += 1
        
        print(f"📊 Результаты проверки:")
        print(f"  • Релевантность резюме: {relevance_resume_score}/100 ({relevance_resume_ok})")
        print(f"  • Релевантность вакансии: {relevance_vacancy_score}/100 ({relevance_vacancy_ok})")
        print(f"  • Конкретность вопросов: {specificity_score}/100 ({specificity_ok})")
        print(f"  • Разнообразие вопросов: {diversity_score}/100 ({diversity_ok})")
        print(f"  • Практическая полезность: {usefulness_score}/100 ({usefulness_ok})")
        print(f"  • Повторений: {repetition_count}")
        print(f"  • Время проверки: {gen_latency} сек")
        
        # Показываем лучшие вопросы
        best_questions = check_result.get("best_questions", [])
        if best_questions:
            print(f"\n🏆 Лучшие вопросы:")
            for j, question in enumerate(best_questions[:3], 1):
                print(f"  {j}. {question}")
        
        # Показываем вопросы для улучшения
        questions_to_improve = check_result.get("questions_to_improve", [])
        if questions_to_improve:
            print(f"\n⚠️ Вопросы для улучшения:")
            for j, question in enumerate(questions_to_improve[:3], 1):
                print(f"  {j}. {question}")
        
        # Сохраняем результат
        all_results.append({
            "id": i,
            "resume_file": os.path.basename(file_path),
            "relevance_to_resume": relevance_resume_ok,
            "relevance_to_resume_score": relevance_resume_score,
            "relevance_to_vacancy": relevance_vacancy_ok,
            "relevance_to_vacancy_score": relevance_vacancy_score,
            "question_specificity": specificity_ok,
            "question_specificity_score": specificity_score,
            "question_diversity": diversity_ok,
            "question_diversity_score": diversity_score,
            "practical_usefulness": usefulness_ok,
            "practical_usefulness_score": usefulness_score,
            "repetition_count": repetition_count,
            "gen_latency": gen_latency,
            "total_tokens": usage.get("total_tokens"),
            "error": None,
            "total_questions": check_result.get("total_questions_analyzed", 0),
            "best_questions": best_questions[:3],
            "questions_to_improve": questions_to_improve[:3]
        })
    
    # Закрываем клиент
    await hr_system.close_client()
    
    # Выводим итоговую статистику
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СТАТИСТИКА ПО ГЕНЕРАЦИИ ВОПРОСОВ")
    print(f"{'='*60}")
    print(f"Всего протестировано: {total} резюме")
    print(f"\n✅ Релевантность резюме: {passed_relevance_resume}/{total}")
    if passed_relevance_resume > 0:
        print(f"   Средняя оценка: {total_relevance_resume/passed_relevance_resume:.1f}/100")
    
    print(f"\n✅ Релевантность вакансии: {passed_relevance_vacancy}/{total}")
    if passed_relevance_vacancy > 0:
        print(f"   Средняя оценка: {total_relevance_vacancy/passed_relevance_vacancy:.1f}/100")
    
    print(f"\n✅ Конкретность вопросов: {passed_specificity}/{total}")
    if passed_specificity > 0:
        print(f"   Средняя оценка: {total_specificity/passed_specificity:.1f}/100")
    
    print(f"\n✅ Разнообразие вопросов: {passed_diversity}/{total}")
    if passed_diversity > 0:
        print(f"   Средняя оценка: {total_diversity/passed_diversity:.1f}/100")
    
    print(f"\n✅ Практическая полезность: {passed_usefulness}/{total}")
    if passed_usefulness > 0:
        print(f"   Средняя оценка: {total_usefulness/passed_usefulness:.1f}/100")
    
    print(f"\n🎯 Все критерии выполнены: {passed_all}/{total}")
    
    # Общая средняя оценка по всем успешным тестам
    successful_tests = [r for r in all_results if r.get("error") is None]
    if successful_tests:
        avg_scores = {
            "relevance_resume": sum(r.get("relevance_to_resume_score", 0) for r in successful_tests) / len(successful_tests),
            "relevance_vacancy": sum(r.get("relevance_to_vacancy_score", 0) for r in successful_tests) / len(successful_tests),
            "specificity": sum(r.get("question_specificity_score", 0) for r in successful_tests) / len(successful_tests),
            "diversity": sum(r.get("question_diversity_score", 0) for r in successful_tests) / len(successful_tests),
            "usefulness": sum(r.get("practical_usefulness_score", 0) for r in successful_tests) / len(successful_tests)
        }
        
        print(f"\n📈 Общие средние оценки:")
        print(f"  • Релевантность резюме: {avg_scores['relevance_resume']:.1f}/100")
        print(f"  • Релевантность вакансии: {avg_scores['relevance_vacancy']:.1f}/100")
        print(f"  • Конкретность: {avg_scores['specificity']:.1f}/100")
        print(f"  • Разнообразие: {avg_scores['diversity']:.1f}/100")
        print(f"  • Полезность: {avg_scores['usefulness']:.1f}/100")
    
    return all_results



#asyncio.run(run_test_extraction())
#asyncio.run(run_test_evaluation())
asyncio.run(run_test_question())
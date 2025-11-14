import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import Gemma3Text

load_dotenv()


def test_interview_generation():
    """Тестируем модуль генерации плана интервью и вопросов"""
    
    # Инициализируем модель
    pipeline = Gemma3Text()
    
    # Создаем папку для результатов
    results_dir = f"ml/evaluation/test_interview_plans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("🚀 Тестируем модуль генерации вопросов для интервью...")
    print(f"📁 Результаты будут сохранены в: {results_dir}")
    
    # Шаг 1: Загружаем или создаем тестовые данные вакансии
    print("\n1. 📋 Подготавливаем данные вакансии...")
    vacancy_data = _create_test_vacancy()
    
    # Сохраняем данные вакансии
    with open(os.path.join(results_dir, "vacancy_data.json"), 'w', encoding='utf-8') as f:
        json.dump(vacancy_data, f, ensure_ascii=False, indent=2)
    print("💾 Сохранены данные вакансии")
    
    # Шаг 2: Загружаем проанализированные резюме
    print("\n2. 📄 Загружаем проанализированные резюме...")
    analyzed_candidates = pipeline._load_and_analyze_resumes(["ml/evaluation/test_analysis_results/resume_ranking_test"])
    
    if not analyzed_candidates:
        print("❌ Нет подходящих резюме для генерации вопросов")
        return
    
    print(f"✅ Загружено {len(analyzed_candidates)} резюме")
    
    # Шаг 3: Генерируем вопросы для каждого кандидата
    print("\n3. ❓ Генерируем вопросы для интервью...")
    interview_results = []
    
    for i, candidate in enumerate(analyzed_candidates, 1):
        print(f"\n--- Кандидат {i}/{len(analyzed_candidates)}: {candidate['filename']} ---")
        
        # Генерируем вопросы для интервью
        interview_plan = pipeline.generate_questions(
            analysis_result=candidate["resume_analysis"],
            vacancy_data=vacancy_data,
            question_type="mixed"
        )
        
        if "error" in interview_plan:
            print(f"❌ Ошибка генерации вопросов: {interview_plan['error']}")
            continue
        
        # Добавляем результат
        candidate_interview_data = {
            "candidate_id": candidate["filename"],
            "resume_analysis": candidate["resume_analysis"],
            "interview_plan": interview_plan,
            "generation_success": True
        }
        
        interview_results.append(candidate_interview_data)
        print(f"✅ Сгенерированы вопросы")
        
        # Сохраняем индивидуальный результат
        _save_individual_interview_result(results_dir, candidate_interview_data, i)
    
    # Шаг 4: Анализируем и ранжируем результаты
    print("\n4. 📊 Анализируем сгенерированные вопросы...")
    analysis_summary = _analyze_interview_results(interview_results)
    
    # Шаг 5: Сохраняем финальные результаты
    print("\n5. 💾 Сохраняем финальные результаты...")
    _save_final_interview_results(results_dir, interview_results, analysis_summary, vacancy_data)
    
    print(f"\n🎯 Тестирование генерации вопросов завершено!")
    print(f"📁 Все результаты сохранены в: {results_dir}")


def _create_test_vacancy():
    """Создает тестовые данные вакансии для Middle Backend Developer"""
    vacancy_data = {
        "job_title": "Middle Backend Developer (Java/Kotlin)",
        "education": "Высшее техническое образование (компьютерные науки, программная инженерия, информационные технологии)",
        "work_experience": 3,
        "desired_salary": 180000,
        "work_schedule": "Полный день",
        "work_format": "Гибридный формат (2-3 дня в офисе, остальное время удаленно)",
        "key_technologies": ["Java", "Kotlin", "Spring Boot", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
        "required_skills": [
            "Разработка микросервисной архитектуры",
            "Оптимизация производительности приложений",
            "Проектирование REST API",
            "Работа с реляционными базами данных",
            "Контейнеризация приложений"
        ],
        "additional_requirements": "Опыт работы с Java 8+, Kotlin, Spring Boot, микросервисной архитектурой, PostgreSQL, Redis, Docker, Kubernetes, знание английского языка на уровне чтения технической документации"
    }
    
    print("📋 Тестовая вакансия Middle Backend Developer:")
    print(f"   • Должность: {vacancy_data['job_title']}")
    print(f"   • Опыт: {vacancy_data['work_experience']} года")
    print(f"   • Ключевые технологии: {', '.join(vacancy_data['key_technologies'][:3])}...")
    
    return vacancy_data

def _analyze_interview_results(interview_results):
    """Анализирует результаты генерации вопросов"""
    successful_generations = [r for r in interview_results if r["generation_success"]]
    failed_generations = [r for r in interview_results if not r["generation_success"]]
    
    # Анализ типов вопросов
    question_categories = {}
    for result in successful_generations:
        plan = result["interview_plan"]
        if isinstance(plan, dict):
            for category in plan.keys():
                question_categories[category] = question_categories.get(category, 0) + 1
    
    summary = {
        "total_candidates": len(interview_results),
        "successful_generations": len(successful_generations),
        "failed_generations": len(failed_generations),
        "success_rate": f"{(len(successful_generations) / len(interview_results) * 100):.1f}%" if interview_results else "0%",
        "question_categories_distribution": question_categories,
        "top_candidates_by_questions": sorted(
            [r for r in successful_generations], 
            reverse=True
        )[:3]
    }
    
    print(f"📊 Анализ результатов генерации:")
    print(f"   • Успешных генераций: {summary['successful_generations']}/{summary['total_candidates']}")
    print(f"   • Среднее количество вопросов: {summary['average_questions_per_candidate']}")
    print(f"   • Категории вопросов: {len(summary['question_categories_distribution'])}")
    
    return summary


def _save_individual_interview_result(results_dir, candidate_data, index):
    """Сохраняет индивидуальный результат генерации вопросов"""
    candidate_dir = os.path.join(results_dir, f"candidate_{index:02d}_{candidate_data['candidate_id'].replace('.json', '')}")
    os.makedirs(candidate_dir, exist_ok=True)
    
    # Сохраняем план интервью
    interview_file = os.path.join(candidate_dir, "interview_plan.json")
    with open(interview_file, 'w', encoding='utf-8') as f:
        json.dump(candidate_data, f, ensure_ascii=False, indent=2)
 
 
def _save_final_interview_results(results_dir, interview_results, analysis_summary, vacancy_data):
    """Сохраняет финальные результаты генерации вопросов"""
    # Создаем сводный отчет
    summary_report = {
        "vacancy_data": vacancy_data,
        "processing_date": datetime.now().isoformat(),
        "generation_summary": analysis_summary,
        "all_candidates": [
            {
                "candidate_id": result["candidate_id"],
                "generation_success": result["generation_success"],
                "interview_plan_keys": list(result["interview_plan"].keys()) if isinstance(result["interview_plan"], dict) else []
            }
            for result in interview_results
        ]
    }
    
    # Сохраняем сводный отчет
    summary_file = os.path.join(results_dir, "interview_generation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Сохранен сводный отчет: interview_generation_summary.json")
    
    # Сохраняем детальные результаты
    details_file = os.path.join(results_dir, "all_interview_plans.json")
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(interview_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Сохранены детальные результаты: all_interview_plans.json")
        


if __name__ == "__main__":
    # Проверяем наличие API ключа
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY не установлен!")
        print("💡 Выполните: export GROQ_API_KEY='ваш-ключ'")
        print("💡 Или создайте файл .env с GROQ_API_KEY=ваш-ключ")
    else:
        test_interview_generation()
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import glob
from typing import Dict, List, Any
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import MistralText
from ml.prompt_templates import UC_MATCHING_PROMPT_WITH_BENCHMARK

load_dotenv()


import json
import os
from datetime import datetime
from pathlib import Path

def evaluate_candidate_with_benchmark(
    pipeline,              # экземпляр MistralText
    resume_json_path,      # путь к JSON файлу с резюме (содержит overall_score)
    job_json_path,         # путь к JSON файлу с вакансией
    criteria_weights=None
):
    """
    Оценивает кандидата с сравнением с эталонной оценкой
    """
    
    try:
        # Загружаем данные из JSON файлов
        with open(resume_json_path, 'r', encoding='utf-8') as f:
            resume_data = json.load(f)
        
        with open(job_json_path, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
        
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения JSON: {e}")
        return None
    
    # Извлекаем benchmark_score из данных резюме
    benchmark_score = resume_data.get('overall_score')
    if benchmark_score is None:
        print(f"В файле {resume_json_path} отсутствует overall_score")
        return None
    
    # Веса по умолчанию
    default_weights = {
        'job_title_weight': 25,
        'education_weight': 15,
        'experience_weight': 20,
        'schedule_weight': 10,
        'format_weight': 10,
        'additional_weight': 25
    }
    
    # Объединяем с переданными весами
    weights = {**default_weights, **(criteria_weights or {})}
    
    try:
        # Используем метод compare_evaluations из pipeline
        result = pipeline.compare_evaluations(
            resume_analysis=resume_data,
            vacancy_data=job_data,
            criteria_weights=weights
        )
        
        # Проверяем на ошибки
        if "error" in result:
            print(f"Ошибка при оценке кандидата: {result['error']}")
            return None
        
        # Добавляем метаданные
        result['evaluation_metadata'] = {
            'candidate_id': Path(resume_json_path).stem,
            'benchmark_used': benchmark_score,
            'criteria_weights': weights,
            'resume_file': resume_json_path,
            'job_file': job_json_path,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"Ошибка при оценке кандидата: {e}")
        return None


def evaluate_all_candidates_in_folder(
    pipeline,              # экземпляр MistralText
    resumes_folder_path,   # путь к папке с JSON файлами резюме
    job_json_path,         # путь к JSON файлу с вакансией
    criteria_weights=None,
    file_extension=".json"
):
    """
    Оценивает всех кандидатов в указанной папке
    """
    
    results = {}
    resumes_folder = Path(resumes_folder_path)
    
    # Проверяем существование папки
    if not resumes_folder.exists():
        print(f"Папка {resumes_folder_path} не существует")
        return results
    
    # Ищем все JSON файлы в папке
    resume_files = list(resumes_folder.glob(f"*{file_extension}"))
    
    if not resume_files:
        print(f"В папке {resumes_folder_path} не найдено файлов с расширением {file_extension}")
        return results
    
    print(f"Найдено {len(resume_files)} файлов для обработки")
    
    for resume_path in resume_files:
        candidate_id = resume_path.stem
        
        print(f"Обрабатывается кандидат: {candidate_id}")
        
        result = evaluate_candidate_with_benchmark(
            pipeline=pipeline,
            resume_json_path=str(resume_path),
            job_json_path=job_json_path,
            criteria_weights=criteria_weights
        )
        
        if result:
            results[candidate_id] = result
            benchmark_used = result['evaluation_metadata']['benchmark_used']
            current_score = result['matching_results']['overall_score']
            difference = result['benchmark_comparison']['score_difference']
            print(f"✅ {candidate_id} - оценка: {current_score} (benchmark: {benchmark_used}, разница: {difference})")
        else:
            print(f"❌ {candidate_id} - ошибка оценки")
    
    return results


# Функция для анализа результатов пакетной обработки
def analyze_batch_results(results):
    """Анализирует результаты пакетной обработки"""
    if not results:
        return {"error": "Нет результатов для анализа"}
    
    analysis = {
        "total_candidates": len(results),
        "successful_evaluations": 0,
        "average_score_difference": 0,
        "consistency_breakdown": {
            "high_consistency": 0,    # расхождение <= 5 баллов
            "medium_consistency": 0,  # расхождение 6-15 баллов  
            "low_consistency": 0      # расхождение > 15 баллов
        },
        "recommendation_breakdown": {
            "recommended": 0,
            "conditionally_recommended": 0,
            "not_recommended": 0
        },
        "candidates_by_priority": {
            "high": [],
            "medium": [],
            "low": []
        },
        "score_ranges": {
            "0-20": 0,
            "21-40": 0,
            "41-60": 0,
            "61-80": 0,
            "81-100": 0
        }
    }
    
    total_difference = 0
    total_current_score = 0
    total_benchmark_score = 0
    
    for candidate_id, result in results.items():
        # Пропускаем результаты с ошибками
        if "error" in result:
            continue
            
        analysis["successful_evaluations"] += 1
        
        try:
            # Преобразуем score_difference в число
            score_diff_str = result['benchmark_comparison']['score_difference']
            # Убираем возможные знаки + и пробелы, преобразуем в число
            score_diff = float(str(score_diff_str).replace('+', '').strip())
            score_diff_abs = abs(score_diff)
            
            total_difference += score_diff_abs
            
            # Также собираем статистику по оценкам
            current_score = float(result['matching_results']['overall_score'])
            benchmark_score = float(result['benchmark_comparison']['benchmark_overall_score'])
            
            total_current_score += current_score
            total_benchmark_score += benchmark_score
            
            # Анализ расхождений
            if score_diff_abs <= 5:
                analysis["consistency_breakdown"]["high_consistency"] += 1
            elif score_diff_abs <= 15:
                analysis["consistency_breakdown"]["medium_consistency"] += 1
            else:
                analysis["consistency_breakdown"]["low_consistency"] += 1
            
            # Анализ диапазонов оценок
            if current_score <= 20:
                analysis["score_ranges"]["0-20"] += 1
            elif current_score <= 40:
                analysis["score_ranges"]["21-40"] += 1
            elif current_score <= 60:
                analysis["score_ranges"]["41-60"] += 1
            elif current_score <= 80:
                analysis["score_ranges"]["61-80"] += 1
            else:
                analysis["score_ranges"]["81-100"] += 1
            
            # Анализ рекомендаций
            recommendation_level = result['recommendation']['level']
            if "рекомендован" in recommendation_level.lower():
                analysis["recommendation_breakdown"]["recommended"] += 1
            elif "условно" in recommendation_level.lower():
                analysis["recommendation_breakdown"]["conditionally_recommended"] += 1
            else:
                analysis["recommendation_breakdown"]["not_recommended"] += 1
            
            # Группировка по приоритету собеседования
            priority = result['recommendation']['interview_priority']
            analysis["candidates_by_priority"][priority].append(candidate_id)
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"Ошибка при анализе кандидата {candidate_id}: {e}")
            print(f"Проблемные данные: {result.get('benchmark_comparison', {})}")
            continue
    
    # Расчет средних значений
    if analysis["successful_evaluations"] > 0:
        analysis["average_score_difference"] = round(total_difference / analysis["successful_evaluations"], 2)
        analysis["average_current_score"] = round(total_current_score / analysis["successful_evaluations"], 2)
        analysis["average_benchmark_score"] = round(total_benchmark_score / analysis["successful_evaluations"], 2)
    
    return analysis


# Функция для сохранения результатов
def save_evaluation_results(results, output_path, include_analysis=True):
    """Сохраняет результаты оценки в JSON файл"""
    try:
        output_data = {
            "evaluation_results": results,
            "summary": {
                "total_evaluated": len(results),
                "evaluation_timestamp": datetime.now().isoformat()
            }
        }
        
        if include_analysis and results:
            output_data["batch_analysis"] = analyze_batch_results(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены в: {output_path}")
        return True
        
    except Exception as e:
        print(f"Ошибка сохранения результатов: {e}")
        return False


# Пример использования
if __name__ == "__main__":
    from ml.models.baseline import MistralText  # импортируйте ваш класс
    
    # Пути к файлам
    job_file = "ml/evaluation/reference_resumes_results/vacancy.json"
    resumes_folder = "ml/evaluation/reference_resumes_evaluation"
    output_file = "ml/evaluation/comparison_results.json"
    
    # Создаем экземпляр pipeline
    pipeline = MistralText()
    
    # Кастомные веса (опционально)
    custom_weights = {
        'job_title_weight': 30,
        'experience_weight': 25,
        'additional_weight': 15
    }
    
    # Пакетная обработка всех кандидатов в папке
    all_results = evaluate_all_candidates_in_folder(
        pipeline=pipeline,
        resumes_folder_path=resumes_folder,
        job_json_path=job_file,
        criteria_weights=None
    )
    
    # Сохранение результатов
    save_evaluation_results(all_results, output_file, include_analysis=True)
    
    # Вывод сводки
    if all_results:
        analysis = analyze_batch_results(all_results)
        print(f"\n📊 Сводка по оценкам:")
        print(f"Обработано кандидатов: {analysis['total_candidates']}")
        print(f"Среднее расхождение: {analysis['average_score_difference']} баллов")
        print(f"Высокая согласованность: {analysis['consistency_breakdown']['high_consistency']} кандидатов")
        print(f"Рекомендовано: {analysis['recommendation_breakdown']['recommended']} кандидатов")
        print(f"Приоритет 'высокий': {len(analysis['candidates_by_priority']['high'])} кандидатов")
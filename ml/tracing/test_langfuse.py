import json
import time
import asyncio
import os
import glob
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import HRBaseline, langfuse

async def test_resume_extraction(test_folders):
    """Тест функции извлечения данных из резюме"""
    hr_system = HRBaseline()
    
    results_dir = f"ml/evaluation/test_extraction_results"

    # Создаем папку для результатов
    os.makedirs(results_dir, exist_ok=True)

    print("🚀 Тестируем baseline интеграцию с реальными файлами...")
    print(f"📁 Результаты будут сохранены в: {results_dir}")
    
    # Тест 1: Поиск тестовых резюме
    print("\n1. 🔍 Ищем тестовые резюме...")
    
    test_files = _find_test_files(test_folders)
    
     # Тест 2: Обработка каждого найденного файла
    print(f"\n2. 📊 Обрабатываем {len(test_files)} файлов...")
    
    successful_analyses = 0
    
    try:
        for i, file_path in enumerate(test_files, 1):
            print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
            
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Анализируем резюме
            extracted_data = await hr_system.extract_data_from_resume(file_path)
            
            if "error" in extracted_data:
                print(f"❌ Ошибка анализа: {extracted_data['error']}")
            else:
                successful_analyses += 1
                print("✅ Анализ завершен успешно!")
                
                # Сохраняем анализ в JSON файл
                json_filename = f"resume_analysis_{file_name}.json"
                json_path = os.path.join(results_dir, json_filename)
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
                    print(f"💾 Сохранен JSON: {json_filename}")
                except Exception as e:
                    print(f"❌ Ошибка сохранения JSON: {e}")
                
        # Небольшая пауза между файлами, чтобы не перегружать систему
        if i < len(test_files):
            await asyncio.sleep(0.1)
    
    finally:
        # Всегда закрываем клиент
        await hr_system.close_client()
        langfuse.flush()

    print(f"\n🎯 Тестирование завершено! Успешно обработано: {successful_analyses}/{len(test_files)} файлов")
    print(f"📁 Все JSON файлы сохранены в: {results_dir}")
    
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

async def test_resume_evaluating(resume_analysis, vacancy_data):
    hr_system = HRBaseline()

    results_dir = f"ml/evaluation/test_evaluating_results"

    # Создаем папку для результатов
    os.makedirs(results_dir, exist_ok=True)
    
    test_files = _find_json_files(resume_analysis)

    # Загружаем данные вакансии (один раз)
    try:
        with open(vacancy_data, 'r', encoding='utf-8') as f:
            vacancy_data = json.load(f)
        print(f"✅ Загружены данные вакансии: {vacancy_data.get('position', 'Не указана')}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных вакансии: {e}")
        return

    successful_analyses = 0
    
    try:
        for i, file_path in enumerate(test_files, 1):
            print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
            
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            # Загружаем анализ резюме из JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    resume_analysis = json.load(f)
                print(f"✅ Загружен анализ резюме ({os.path.getsize(file_path)/1024:.1f} KB)")
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка парсинга JSON: {e}")
                continue
            except Exception as e:
                print(f"❌ Ошибка загрузки файла: {e}")
                continue

            # Анализируем резюме
            extracted_data = await hr_system.evaluate_candidate_match(resume_analysis, vacancy_data)
            
            if "error" in extracted_data:
                print(f"❌ Ошибка анализа: {extracted_data['error']}")
            else:
                successful_analyses += 1
                print("✅ Анализ завершен успешно!")
                
                # Сохраняем анализ в JSON файл
                json_filename = f"resume_analysis_{file_name}.json"
                json_path = os.path.join(results_dir, json_filename)
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
                    print(f"💾 Сохранен JSON: {json_filename}")
                except Exception as e:
                    print(f"❌ Ошибка сохранения JSON: {e}")
                
        # Небольшая пауза между файлами, чтобы не перегружать систему
        if i < len(test_files):
            await asyncio.sleep(0.1)
    
    finally:
        # Всегда закрываем клиент
        await hr_system.close_client()
        langfuse.flush()

def _find_json_files(folders):
    """Находит все JSON файлы в указанных папках"""
    json_files = []
    
    if isinstance(folders, str):
        folders = [folders]
    
    for folder in folders:
        if os.path.exists(folder):
            # Ищем все JSON файлы в папке
            pattern = os.path.join(folder, "*.json")
            found_files = glob.glob(pattern)
            
            for file_path in found_files:
                print(f"📄 Найден файл: {os.path.basename(file_path)}")
                json_files.append(file_path)
        else:
            print(f"⚠️ Папка не найдена: {folder}")
    
    return json_files

async def test_question_generation(resume_analysis_path, vacancy_data_path, output_folder="ml/evaluation/test_questions_results"):
    """Тест функции генерации вопросов для кандидатов"""
    hr_system = HRBaseline()
    
    results_dir = output_folder

    # Создаем папку для результатов
    os.makedirs(results_dir, exist_ok=True)

    print("🚀 Тестируем генерацию вопросов для кандидатов...")
    print(f"📁 Результаты будут сохранены в: {results_dir}")
    
    # Ищем JSON файлы с анализами резюме
    print("\n1. 🔍 Ищем файлы с анализами резюме...")
    
    json_files = _find_json_files(resume_analysis_path)
    
    if not json_files:
        print("❌ Не найдено JSON файлов для тестирования")
        return
    
    # Загружаем данные вакансии
    print("\n2. 📋 Загружаем данные вакансии...")
    
    try:
        with open(vacancy_data_path, 'r', encoding='utf-8') as f:
            vacancy_data = json.load(f)
        print(f"✅ Загружены данные вакансии: {vacancy_data.get('position', 'Не указана')}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных вакансии: {e}")
        return
    
    # Тест 3: Генерация вопросов для каждого кандидата
    print(f"\n3. ❓ Генерируем вопросы для {len(json_files)} кандидатов...")
    
    successful_generations = 0
    
    try:
        for i, file_path in enumerate(json_files, 1):
            print(f"\n{'='*60}")
            print(f"📄 Кандидат {i}/{len(json_files)}: {os.path.basename(file_path)}")
            print(f"{'='*60}")
            
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Загружаем анализ резюме
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    resume_analysis = json.load(f)
                
                print(f"✅ Загружен анализ резюме")
                
                # Показываем краткую информацию о кандидате
                if isinstance(resume_analysis, dict):
                    candidate_name = resume_analysis.get('name') or resume_analysis.get('candidate_name', 'Не указано')
                    candidate_position = resume_analysis.get('position') or resume_analysis.get('desired_position', 'Не указано')
                    print(f"👤 Кандидат: {candidate_name}")
                    print(f"💼 Позиция: {candidate_position}")
                    
            except Exception as e:
                print(f"❌ Ошибка загрузки JSON: {e}")
                continue
            
            # Генерируем вопросы
            print("\n🎯 Генерируем вопросы для собеседования...")
            
            start_time = time.time()
            
            try:
                # Пробуем разные методы генерации вопросов
                questions_data = None

                questions_data = await hr_system.generate_interview_questions(resume_analysis, vacancy_data)

                generation_time = time.time() - start_time
                
                if questions_data is None:
                    print("❌ Не удалось сгенерировать вопросы")
                    continue
                
                # Проверяем результат
                if isinstance(questions_data, dict) and "error" in questions_data:
                    print(f"❌ Ошибка генерации: {questions_data['error']}")
                else:
                    successful_generations += 1
                    print(f"✅ Вопросы сгенерированы успешно! Время: {generation_time:.2f} сек")
                    
                    # Анализируем и выводим информацию о вопросах
                    if isinstance(questions_data, dict):
                        questions_list = questions_data.get('questions', [])
                        if isinstance(questions_list, list):
                            print(f"📝 Сгенерировано вопросов: {len(questions_list)}")
                            
                            # Выводим категории вопросов, если есть
                            if 'categories' in questions_data:
                                categories = questions_data.get('categories', {})
                                print(f"📊 Категории вопросов:")
                                for category, count in categories.items():
                                    print(f"   • {category}: {count} вопросов")
                            
                            # Выводим первые 3 вопроса
                            if questions_list and len(questions_list) > 0:
                                print("\n🎤 Примеры вопросов:")
                                for j, question in enumerate(questions_list[:3], 1):
                                    if isinstance(question, dict):
                                        q_text = question.get('question', str(question))
                                        q_type = question.get('type', 'Общий')
                                        print(f"   {j}. [{q_type}] {q_text[:80]}...")
                                    else:
                                        print(f"   {j}. {str(question)[:80]}...")
                        else:
                            print(f"⚠️ Вопросы в неожиданном формате: {type(questions_list).__name__}")
                    
                    # Сохраняем вопросы в JSON файл
                    timestamp = int(time.time())
                    json_filename = f"questions_{file_name}_{timestamp}.json"
                    json_path = os.path.join(results_dir, json_filename)
                    
                    # Создаем полную запись с метаданными
                    full_result = {
                        'source_file': file_path,
                        'candidate_name': candidate_name,
                        'candidate_position': candidate_position,
                        'vacancy_position': vacancy_data.get('position'),
                        'vacancy_company': vacancy_data.get('company'),
                        'generation_time_seconds': generation_time,
                        'timestamp': timestamp,
                        'questions_data': questions_data,
                        'metadata': {
                            'total_questions': len(questions_list) if isinstance(questions_list, list) else 0,
                            'candidate_skills': resume_analysis.get('skills', [])[:5] if isinstance(resume_analysis.get('skills'), list) else [],
                            'vacancy_requirements': vacancy_data.get('requirements', [])[:5] if isinstance(vacancy_data.get('requirements'), list) else []
                        }
                    }
                    
                    try:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(full_result, f, ensure_ascii=False, indent=2)
                        print(f"💾 Сохранены вопросы: {json_filename}")
                    except Exception as e:
                        print(f"❌ Ошибка сохранения JSON: {e}")
                
            except Exception as e:
                print(f"❌ Ошибка при генерации вопросов: {e}")
                import traceback
                traceback.print_exc()
            
            # Пауза между кандидатами
            if i < len(json_files):
                await asyncio.sleep(0.1)
    
    finally:
        # Всегда закрываем клиент
        await hr_system.close_client()
        langfuse.flush()
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СТАТИСТИКА ГЕНЕРАЦИИ ВОПРОСОВ")
    print(f"{'='*60}")
    print(f"Всего кандидатов: {len(json_files)}")
    print(f"✅ Успешно сгенерированы вопросы: {successful_generations}")
    print(f"❌ Не удалось сгенерировать: {len(json_files) - successful_generations}")
    
    if successful_generations > 0:
        # Создаем сводный отчет
        summary_filename = f"questions_summary_{int(time.time())}.json"
        summary_path = os.path.join(results_dir, summary_filename)
        
        summary = {
            'test_type': 'question_generation',
            'total_candidates': len(json_files),
            'successful_generations': successful_generations,
            'failed_generations': len(json_files) - successful_generations,
            'vacancy_position': vacancy_data.get('position'),
            'test_timestamp': int(time.time()),
            'results_folder': results_dir
        }
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n📋 Сводный отчет сохранен: {summary_filename}")
        except Exception as e:
            print(f"❌ Ошибка сохранения сводного отчета: {e}")
    
    print(f"\n🎯 Генерация вопросов завершена!")
    print(f"📁 Все файлы сохранены в: {results_dir}")

if __name__ == "__main__":

    test_folders = [
        "data/test_resumes"
    ]

    vacancy_data = "C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/vacancy.json"

    resume_analysis = f"ml/evaluation/test_extraction_results"

    
    asyncio.run(test_resume_extraction(test_folders))

    #asyncio.run(test_resume_evaluating(resume_analysis, vacancy_data))

    #asyncio.run(test_question_generation(resume_analysis, vacancy_data))


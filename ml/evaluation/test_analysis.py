import sys
import os
import glob
import json
from dotenv import load_dotenv

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import Gemma3Text

load_dotenv()

def test_baseline_integration():
    """Тестируем интеграцию с baseline.py с реальными файлами"""
    
    # Инициализируем модель и парсер
    pipeline = Gemma3Text()
    
    # Создаем папку для результатов
    results_dir = f"ml/evaluation/test_analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("🚀 Тестируем baseline интеграцию с реальными файлами...")
    print(f"📁 Результаты будут сохранены в: {results_dir}")
    
    # Тест 1: Поиск тестовых резюме
    print("\n1. 🔍 Ищем тестовые резюме...")
    
    test_files = _find_test_files()
    
    if not test_files:
        print("❌ Тестовые файлы не найдены.")
    else:
        print(f"✅ Найдено {len(test_files)} тестовых файлов")
    
    # Тест 2: Обработка каждого найденного файла
    print(f"\n2. 📊 Обрабатываем {len(test_files)} файлов...")
    
    successful_analyses = 0
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n--- Файл {i}/{len(test_files)}: {os.path.basename(file_path)} ---")
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Анализируем резюме
        analysis_result = pipeline.analyze_resume(file_path)
        
        if "error" in analysis_result:
            print(f"❌ Ошибка анализа: {analysis_result['error']}")
        else:
            successful_analyses += 1
            print("✅ Анализ завершен успешно!")
            
            # Сохраняем анализ в JSON файл
            json_filename = f"resume_analysis_{file_name}.json"
            json_path = os.path.join(results_dir, json_filename)
            
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                print(f"💾 Сохранен JSON: {json_filename}")
            except Exception as e:
                print(f"❌ Ошибка сохранения JSON: {e}")
            
    
    print(f"\n🎯 Тестирование завершено! Успешно обработано: {successful_analyses}/{len(test_files)} файлов")
    print(f"📁 Все JSON файлы сохранены в: {results_dir}")


def _find_test_files():
    """Ищет тестовые файлы в различных папках"""
    test_folders = [
        "data/test_resumes"
    ]
    
    test_files = []
    for folder in test_folders:
        if os.path.exists(folder):
            # Ищем все поддерживаемые форматы
            patterns = [f"{folder}/*.pdf", f"{folder}/*.docx", f"{folder}/*.doc", f"{folder}/*.txt"]
            for pattern in patterns:
                found_files = glob.glob(pattern)
                test_files.extend(found_files)
    
    return test_files


if __name__ == "__main__":
    # Проверяем наличие API ключа
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY не установлен!")
        print("💡 Выполните: export GROQ_API_KEY='ваш-ключ'")
        print("💡 Или создайте файл .env с GROQ_API_KEY=ваш-ключ")
    else:
        # Запускаем оба теста
        test_baseline_integration()
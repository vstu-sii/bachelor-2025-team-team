import os
import sys
import glob
from datetime import datetime

sys.path.append('.')

from ml.utils.file_parser import FileParser


def test_parser():
    """Тестируем парсер на реальных файлах с сохранением только extracted_text.txt"""
    
    # Создаем папку для результатов
    results_dir = _create_parser_results_directory()
    print(f"📁 Результаты будут сохранены в: {results_dir}")
    
    # Тестовые файлы
    test_files = glob.glob("data/*")
    successful_parses = 0
    total_files = len(test_files)
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n🧪 Тестируем: {file_path}")
            
            # Создаем папку для этого файла
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_results_dir = os.path.join(results_dir, file_name)
            os.makedirs(file_results_dir, exist_ok=True)
            
            try:
                # Парсим файл
                text = FileParser.extract_text_from_file(file_path)
                cleaned = FileParser.clean_extracted_text(text)
                
                # Сохраняем только extracted_text.txt
                _save_extracted_text(file_results_dir, cleaned, file_path)
                
                if FileParser.is_successful_extraction(text):
                    successful_parses += 1
                    print(f"✅ Успех! Извлечено {len(cleaned)} символов")
                    print(f"📝 Предпросмотр: {cleaned[:200]}...")
                else:
                    print(f"⚠️ Проблема: {text}")
                    
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                _save_error_file(file_results_dir, file_path, str(e))
        else:
            print(f"⚠️ Файл не найден: {file_path}")
    
    print(f"\n🎯 Парсинг завершен! Успешно: {successful_parses}/{total_files} файлов")
    print(f"📁 Все тексты сохранены в: {results_dir}")


def _create_parser_results_directory():
    """Создает папку для результатов парсинга"""
    results_dir = f"ml/utils/test_parser_results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _save_extracted_text(results_dir: str, cleaned_text: str, file_path: str):
    """Сохраняет извлеченный текст в файл"""
    try:
        text_file_path = os.path.join(results_dir, "extracted_text.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            # Добавляем заголовок с информацией о файле
            header = f"Файл: {os.path.basename(file_path)}\n"
            header += f"Время извлечения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            header += f"Длина текста: {len(cleaned_text)} символов\n"
            header += "=" * 50 + "\n\n"
            f.write(header + cleaned_text)
        print(f"   💾 Сохранен: extracted_text.txt ({len(cleaned_text)} символов)")
        
    except Exception as e:
        print(f"   ❌ Ошибка сохранения текста: {e}")


def _save_error_file(results_dir: str, file_path: str, error_message: str):
    """Сохраняет информацию об ошибке в текстовый файл"""
    try:
        error_path = os.path.join(results_dir, "error.txt")
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Ошибка при обработке файла: {file_path}\n")
            f.write(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ошибка: {error_message}\n")
        print(f"   💾 Сохранен: error.txt")
    except Exception as e:
        print(f"   ❌ Ошибка сохранения error: {e}")


if __name__ == "__main__":
    print("🚀 Запуск теста парсера файлов...")
    test_parser()
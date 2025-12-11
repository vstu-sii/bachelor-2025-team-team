# ml/utils/file_parser.py
import os
import pdfplumber
import subprocess
import tempfile

class FileParser:
    """Парсер для извлечения текста из файлов"""
    
    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        """Извлекает текст из файла по расширению"""
        if not os.path.exists(file_path):
            return f"Файл не найден: {file_path}"
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return FileParser._extract_from_pdf(file_path)
        elif ext == '.docx':
            return FileParser._extract_from_docx(file_path)
        elif ext == '.doc':
            return FileParser._extract_from_doc(file_path)
        elif ext == '.txt':
            return FileParser._extract_from_txt(file_path)
        else:
            return f"Неподдерживаемый формат файла: {ext}"
    
    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        """Извлечение текста из DOCX"""
        try:
            # Пробуем разные способы
            
            # Способ 1: python-docx
            try:
                import docx
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text += paragraph.text + "\n"
                if text.strip():
                    return text.strip()
            except ImportError:
                pass
            
            # Способ 2: Конвертация через LibreOffice
            return FileParser._convert_with_libreoffice(file_path)
            
        except Exception as e:
            return f"Ошибка чтения DOCX: {str(e)}"
    
    @staticmethod
    def _extract_from_doc(file_path: str) -> str:
        """Извлечение текста из DOC"""
        # Пробуем win32com (только Windows)
        try:
            return FileParser._extract_with_win32com(file_path)
        except Exception as e:
            return f"Ошибка чтения DOC: {str(e)}. Установите Microsoft Word или используйте DOCX/TXT"
    
    @staticmethod
    def _extract_with_win32com(file_path: str) -> str:
        """Извлечение через Microsoft Word (только Windows)"""
        try:
            import win32com.client
            
            # Создаем COM объект Word
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False  # Скрываем окно Word
            
            try:
                # Открываем документ
                doc = word.Documents.Open(os.path.abspath(file_path))
                
                # Извлекаем весь текст
                text = doc.Content.Text
                
                # Закрываем документ и приложение
                doc.Close(SaveChanges=False)
                word.Quit()
                
                return text.strip() if text.strip() else "DOC файл пуст"
                
            except Exception as e:
                word.Quit()
                return f"Ошибка Word COM: {str(e)}"
                
        except ImportError:
            return "Для чтения DOC установите: pip install pywin32 и Microsoft Word"
        except Exception as e:
            return f"Ошибка win32com: {str(e)}"

    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Извлечение текста из PDF"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip() if text.strip() else "Не удалось извлечь текст из PDF"
        except Exception as e:
            return f"Ошибка чтения PDF: {str(e)}"
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Извлечение текста из TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp1251') as f:
                    return f.read().strip()
            except:
                return "Не удалось декодировать текстовый файл"
        except Exception as e:
            return f"Ошибка чтения TXT: {str(e)}"

    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """Очистка извлеченного текста"""
        # Удаляем лишние переносы строк
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Объединяем короткие строки (вероятно часть одного предложения)
        cleaned_lines = []
        for line in lines:
            if len(line) < 50 and cleaned_lines:
                cleaned_lines[-1] += " " + line
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def is_successful_extraction(text: str) -> bool:
        """Проверяет, успешно ли извлечен текст"""
        error_indicators = [
            "Файл не найден",
            "Не удалось", 
            "Ошибка",
            "Неподдерживаемый",
            "установите",
            "конвертируйте",
            "пуст",
            "декодировать",
            "📝",
            "🔧"
        ]
        return not any(indicator in text.lower() for indicator in error_indicators)
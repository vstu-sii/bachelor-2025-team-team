import os
import json
import re
import sys
from groq import Groq
from langchain_groq import ChatGroq
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.prompt_templates import UC_ANALYSIS_PROMPT, UC_MATCHING_PROMPT, UC_QUESTION_GENERATION_PROMPT
from ml.utils.file_parser import FileParser


class Gemma3Text:
    """Основной класс для работы с текстовой моделью - аналог из примера"""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        #self.model = "llama-3.3-70b-versatile"  # Актуальная модель
        self.model = "openai/gpt-oss-120b"
        
        # LangChain версия для промптов
        self.llm = ChatGroq(
            model=self.model,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )

    def analyze_resume(self, file_path: str = None):
        """Анализ резюме"""
        start_time = time.time()
        try:
            # Парсим файл
            print(f"📄 Парсим файл...")
            resume_text = FileParser.extract_text_from_file(file_path)
            
            # Проверяем успешность извлечения
            if not FileParser.is_successful_extraction(resume_text):
                print(f"❌ Ошибка парсинга: {resume_text}")
                return result
            
            # Очищаем текст
            cleaned_text = FileParser.clean_extracted_text(resume_text)
            print(f"✅ Извлечено {len(cleaned_text)} символов")
            
            # Анализируем резюме
            print("🔍 Анализируем резюме...")
            
            # Используем промпт-шаблон
            chain = UC_ANALYSIS_PROMPT | self.llm
            response = chain.invoke({"resume_text": cleaned_text})
            
            result = self._parse_json_response(response.content)
                        
            if "error" not in result:
                print("✅ Анализ завершен успешно")
            else:
                print("❌ Ошибка анализа:", result["error"])
                
            return result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")

            return {"error": error_msg}
    
    def match_vacancy_with_resume(self, resume_analysis: dict, vacancy_data: dict):
        """Сопоставляет требования вакансии с данными кандидата"""
        
        start_time = time.time()
        
        try:
            print("🎯 Сопоставляем вакансию с резюме...")
            
            # Используем LangChain если доступен
            if self.llm and UC_MATCHING_PROMPT:
                chain = UC_MATCHING_PROMPT | self.llm
                response = chain.invoke({
                    "job_title": vacancy_data.get("job_title", ""),
                    "education": vacancy_data.get("education", ""),
                    "work_experience": vacancy_data.get("work_experience", 0),
                    "desired_salary": vacancy_data.get("desired_salary", 0),
                    "work_schedule": vacancy_data.get("work_schedule", ""),
                    "work_format": vacancy_data.get("work_format", ""),
                    "additional_requirements": vacancy_data.get("additional_requirements", ""),
                    "resume_analysis": str(resume_analysis)
                })
                result_text = response.content
            else:
                # Fallback: прямой вызов Groq API
                prompt = self._create_matching_prompt(resume_analysis, vacancy_data)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
            
            result = self._parse_json_response(result_text)
            
            return result
            
        except Exception as e:
            return {"error": f"Matching failed: {str(e)}"}
    
   
    def generate_questions(self, analysis_result: dict, vacancy_data: dict, question_type: str = "mixed"):
        """Генерация вопросов для интервью"""

        start_time = time.time()

        try:
            print("❓ Генерируем вопросы для интервью...")
            
            chain = UC_QUESTION_GENERATION_PROMPT | self.llm
            response = chain.invoke({
                "resume_analysis": str(analysis_result),
                "vacancy_requirements": vacancy_data,
                "skill_gaps": analysis_result.get("match_analysis", {}).get("gaps", []),
                "strengths": analysis_result.get("match_analysis", {}).get("strengths", []),
                "experience_summary": f"{analysis_result.get('experience', {}).get('total_years', 0)} лет опыта",
                "position_level": "middle",
                "additional_instructions": ""
            })
            
            result = self._parse_json_response(response.content)

            if "error" not in result:
                print("✅ Вопросы сгенерированы успешно")
            else:
                print("❌ Ошибка генерации:", result["error"])
                
            return result
            
        except Exception as e:
            error_msg = f"Question generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def _parse_json_response(self, response_text: str):
        """Улучшенный парсинг JSON ответа от модели"""
        try:
            # Убираем возможные markdown блоки кода
            cleaned_text = response_text.strip()
            
            # Удаляем ```json и ``` markers
            cleaned_text = re.sub(r'```json|```', '', cleaned_text).strip()
            
            # Ищем JSON в тексте (на случай если модель добавила текст)
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)
            
            print(f"🔧 Очищенный текст для парсинга: {cleaned_text[:200]}...")
            
            # Парсим JSON
            result = json.loads(cleaned_text)
            print("✅ JSON успешно распарсен")
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"📄 Проблемный текст: {response_text}")
            
            # Пробуем исправить common JSON errors
            fixed_text = self._fix_json_errors(response_text)
            if fixed_text != response_text:
                print("🛠️ Пробуем исправить JSON...")
                try:
                    result = json.loads(fixed_text)
                    print("✅ JSON исправлен и распарсен")
                    return result
                except json.JSONDecodeError:
                    pass
            
            return {
                "error": f"JSON parse error: {str(e)}",
                "raw_response": response_text
            }
        except Exception as e:
            return {
                "error": f"Unexpected parse error: {str(e)}",
                "raw_response": response_text
            }
    
    def _fix_json_errors(self, text: str) -> str:
        """Пытается исправить common JSON ошибки"""
        fixed = text.strip()
        
        # Удаляем ```json и ``` markers
        fixed = re.sub(r'```json|```', '', fixed).strip()
        
        # Ищем JSON объект
        json_match = re.search(r'\{.*\}', fixed, re.DOTALL)
        if json_match:
            fixed = json_match.group(0)
        
        # Исправляем незакрытые кавычки
        fixed = re.sub(r'([^"]|^)"([^",}\]\s]*)', r'\1"\2"', fixed)
        
        # Исправляем trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Исправляем одинарные кавычки в двойные (для JSON)
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
        
        return fixed
    
    def _load_and_analyze_resumes(self, test_folders):
        """Загружает резюме из папки"""
        
        analyzed_candidates = []
        
        for folder in test_folders:
            if os.path.exists(folder):
                # Ищем файлы резюме
                import glob
                patterns = [f"{folder}/*.json"]
                
                for pattern in patterns:
                    found_files = glob.glob(pattern)
                    
                    for file_path in found_files:
                        print(f"📄 Добавляем: {os.path.basename(file_path)}")

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                analysis_result = json.load(f)
                        except Exception as e:
                            print(f"❌ Ошибка загрузки JSON: {e}")
                            return None

                        # Добавляем кандидата
                        candidate_data = {
                            "filename": os.path.basename(file_path),
                            "file_path": file_path,
                            "resume_analysis": analysis_result
                        }
                        
                        analyzed_candidates.append(candidate_data)
                        print(f"   ✅ Успешно добавлено")
                        

        if not analyzed_candidates:
            print("⚠️ Файлы не найдены")
        
        return analyzed_candidates
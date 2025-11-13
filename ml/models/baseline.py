import os
import json
import re
import sys
import time
import httpx
import requests
import glob
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.prompt_templates import UC_ANALYSIS_PROMPT, UC_MATCHING_PROMPT, UC_QUESTION_GENERATION_PROMPT, UC_MATCHING_PROMPT_WITH_BENCHMARK
from ml.utils.file_parser import FileParser

# Загружаем переменные окружения
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small"  # Изменили на Mistral

MAX_RETRIES = 3
RETRY_DELAY = 2  # секунды

def _sanitize_json_string(s: str) -> str:
    """Удаляем управляющие символы, которые ломают JSON."""
    return re.sub(r'[\x00-\x1f\x7f]', ' ', s)

class MistralText:
    """Основной класс для работы с Mistral моделью"""
    
    def __init__(self):
        self.api_key = MISTRAL_API_KEY
        self.model = MISTRAL_MODEL
        self.url = MISTRAL_URL

    def _call_mistral_api(self, prompt_text: str) -> dict:
        """Синхронный вызов Mistral API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    return self._parse_json_response(content)
                else:
                    if attempt == MAX_RETRIES:
                        return {"error": f"Mistral API error: {response.status_code}"}
                    time.sleep(RETRY_DELAY)
                    
            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"error": f"Mistral API call failed: {str(e)}"}
                time.sleep(RETRY_DELAY)
        
        return {"error": "Failed after retries"}

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
                return {"error": f"Parsing failed: {resume_text}"}
            
            # Очищаем текст
            cleaned_text = FileParser.clean_extracted_text(resume_text)
            print(f"✅ Извлечено {len(cleaned_text)} символов")
            
            # Анализируем резюме
            print("🔍 Анализируем резюме...")
            
            # Формируем промпт для Mistral
            prompt_text = UC_ANALYSIS_PROMPT.format_messages(resume_text=cleaned_text)
            prompt_text = "\n".join([m.content for m in prompt_text])
            
            # Вызываем Mistral API
            result = self._call_mistral_api(prompt_text)
                        
            if "error" not in result:
                print("✅ Анализ завершен успешно")
            else:
                print("❌ Ошибка анализа:", result["error"])
                
            return result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def match_vacancy_with_resume(self, resume_analysis: dict, vacancy_data: dict, criteria_weights: dict = None, candidate_id: str = None):
        """Сопоставляет требования вакансии с данными кандидата"""
        
        start_time = time.time()
        
        try:
            print("🎯 Сопоставляем вакансию с резюме...")

            # Веса по умолчанию
            default_weights = {
                'job_title_weight': 20,
                'education_weight': 15,
                'experience_weight': 20,
                'schedule_weight': 10,
                'format_weight': 10,
                'additional_weight': 25
            }
            
            # Объединяем с переданными весами
            weights = {**default_weights, **(criteria_weights or {})}
            
            # Генерируем candidate_id если не передан
            if candidate_id is None:
                candidate_id = f"candidate_{int(time.time())}"

            # Формируем промпт для Mistral
            prompt_text = UC_MATCHING_PROMPT.format_messages(
                candidate_id=candidate_id,
                job_title=vacancy_data.get("job_title", ""),
                education=vacancy_data.get("education", ""),
                work_experience=vacancy_data.get("work_experience", 0),
                desired_salary=vacancy_data.get("desired_salary", 0),
                work_schedule=vacancy_data.get("work_schedule", ""),
                work_format=vacancy_data.get("work_format", ""),
                additional_requirements=vacancy_data.get("additional_requirements", ""),
                resume_analysis=str(resume_analysis),
                job_title_weight=weights['job_title_weight'],
                education_weight=weights['education_weight'],
                experience_weight=weights['experience_weight'],
                schedule_weight=weights['schedule_weight'],
                format_weight=weights['format_weight'],
                additional_weight=weights['additional_weight']
            )
            
            prompt_text = "\n".join([m.content for m in prompt_text])
            
            # Вызываем Mistral API
            result = self._call_mistral_api(prompt_text)
            
            return result
            
        except Exception as e:
            return {"error": f"Matching failed: {str(e)}"}
    
    def generate_questions(self, analysis_result: dict, vacancy_data: dict, question_type: str = "mixed"):
        """Генерация вопросов для интервью"""

        start_time = time.time()

        try:
            print("❓ Генерируем вопросы для интервью...")
            
            # Формируем промпт для Mistral
            prompt_text = UC_QUESTION_GENERATION_PROMPT.format_messages(
                resume_analysis=str(analysis_result),
                vacancy_requirements=str(vacancy_data),
                additional_instructions="Сгенерируй разнообразные вопросы, охватывающие все аспекты кандидата",
                skill_gaps=", ".join(analysis_result.get("match_analysis", {}).get("gaps", [])),
                strengths=", ".join(analysis_result.get("match_analysis", {}).get("strengths", [])),
                experience_summary=f"{analysis_result.get('experience', {}).get('total_years', 0)} лет опыта"
            )
            
            prompt_text = "\n".join([m.content for m in prompt_text])
            
            # Вызываем Mistral API
            result = self._call_mistral_api(prompt_text)

            if "error" not in result:
                print("✅ Вопросы сгенерированы успешно")
            else:
                print("❌ Ошибка генерации:", result["error"])
                
            return result
            
        except Exception as e:
            error_msg = f"Question generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def compare_evaluations(self, resume_analysis: dict, vacancy_data: dict, criteria_weights: dict = None):
        """
        Сравнивает оценку кандидата с эталонной оценкой
        """
        
        start_time = time.time()
        
        try:
            # Извлекаем benchmark_score из данных резюме
            benchmark_score = resume_analysis.get('overall_score')
            if benchmark_score is None:
                return {"error": "В данных резюме отсутствует overall_score"}
            
             # Веса по умолчанию
            default_weights = {
                'job_title_weight': 0.25,
                'education_weight': 0.15,
                'experience_weight': 0.30,
                'schedule_weight': 0.10,
                'format_weight': 0.10,
                'additional_weight': 0.10
            }
            
            # Объединяем с переданными весами
            weights = {**default_weights, **(criteria_weights or {})}

            # Формируем промпт для Mistral
            prompt_text = UC_MATCHING_PROMPT_WITH_BENCHMARK.format_messages(
                job_title=vacancy_data.get("job_title", ""),
                education=vacancy_data.get("education", ""),
                work_experience=vacancy_data.get("work_experience", 0),
                desired_salary=vacancy_data.get("desired_salary", 0),
                work_schedule=vacancy_data.get("work_schedule", ""),
                work_format=vacancy_data.get("work_format", ""),
                additional_requirements=vacancy_data.get("additional_requirements", ""),
                benchmark_analysis=str(resume_analysis),
                resume_analysis=json.dumps(resume_analysis, ensure_ascii=False),
                benchmark_score=benchmark_score,
                job_title_weight=weights['job_title_weight'],
                education_weight=weights['education_weight'],
                experience_weight=weights['experience_weight'],
                schedule_weight=weights['schedule_weight'],
                format_weight=weights['format_weight'],
                additional_weight=weights['additional_weight']
            )
            
            prompt_text = "\n".join([m.content for m in prompt_text])
            
            # Вызываем Mistral API
            result = self._call_mistral_api(prompt_text)
            
            # Добавляем метаданные
            if "error" not in result:
                result["evaluation_metadata"] = {
                    "benchmark_used": benchmark_score,
                    "criteria_weights": weights,
                    "processing_time": time.time() - start_time,
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Matching failed: {str(e)}"}

    def _parse_json_response(self, text: str) -> dict:
        """Парсинг JSON ответа от модели"""
        try:
            # Чистим Markdown
            clean = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
            clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)
            
            json_start = clean.find('{')
            json_end = clean.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                clean = clean[json_start:json_end]
            
            clean = _sanitize_json_string(clean)
            return json.loads(clean)
            
        except Exception as e:
            return {"error": f"JSON parsing failed: {str(e)}", "raw_output": text}

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


# Пример использования
if __name__ == "__main__":
    mistral = MistralText()
    results = []
    # Пример данных
    resume_folder = "ml/evaluation/test_analysis_results/resume_ranking_test"
    vacancy_path = "ml/evaluation/reference_resumes_results/vacancy.json"
    with open(vacancy_path, 'r', encoding='utf-8') as f:
        vacancy_data = json.load(f)
   
    # Тестируем оценку кандидата
    print("🧪 Testing Mistral HR system...")

    # Ищем все JSON файлы в папке
    pattern = os.path.join(resume_folder, "*.json")
    resume_files = glob.glob(pattern)

    for i, resume_file in enumerate(resume_files, 1):
        try:
            print(f"\n--- Обрабатываем резюме {i}/{len(resume_files)}: {os.path.basename(resume_file)} ---")
            
            # Загружаем резюме из файла
            with open(resume_file, 'r', encoding='utf-8') as f:
                resume_data = json.load(f)
            
            result = mistral.match_vacancy_with_resume(resume_data, vacancy_data)

            # Сохраняем результат
            result_info = {
                "resume_file": resume_file,
                "candidate_id": os.path.basename(resume_file),
                "result": result
            }
            results.append(result_info)
            
            
        except Exception as e:
            print(f"❌ Ошибка обработки {resume_file}: {e}")
            results.append({
                "resume_file": resume_file,
                "error": str(e)
            })
    
    for result in results:
        print(f"\n📊 Результат для {result['candidate_id']}:")
        if 'result' in result:
            score = result['result'].get('matching_results', {}).get('overall_score', 'N/A')
            print(f"   Оценка: {score}/100")
        else:
            print(f"   Ошибка: {result['error']}")

    print("✅ Result:", json.dumps(result, ensure_ascii=False, indent=2))
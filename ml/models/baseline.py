import os
import re
import json
import time
import sys
import asyncio
import httpx
from dotenv import load_dotenv
from langfuse import Langfuse, observe  
from transformers import AutoTokenizer  

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.prompt_templates import ( 
    UC_MATCHING_PROMPT, 
    UC_QUESTION_GENERATION_PROMPT,
    UC_ANALYSIS_PROMPT
)
from ml.utils.file_parser import FileParser
# Загружаем переменные окружения
load_dotenv()

# Инициализируем Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Токенайзер для подсчета токенов
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-medium-2508"

MAX_RETRIES = 3
RETRY_DELAY = 2  # секунды

def _sanitize_json_string(s: str) -> str:
    """Удаляем управляющие символы, которые ломают JSON."""
    return re.sub(r'[\x00-\x1f\x7f]', ' ', s)

def count_tokens_and_cost(prompt: str, output: str, model: str = "mistral-small"):
    """Подсчет токенов и стоимости"""
    input_tokens = len(tokenizer.encode(prompt))
    output_tokens = len(tokenizer.encode(output))
    total_tokens = input_tokens + output_tokens

    # Тарифы для Mistral (примерные)
    price_per_input = 0.4 / 1_000_000
    price_per_output = 2 / 1_000_000
    cost = input_tokens * price_per_input + output_tokens * price_per_output

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
    }, cost

class HRBaseline:
    """Бейслайн система для HR-оценки кандидатов"""
    
    def __init__(self):
        self.api_key = MISTRAL_API_KEY
        self.model = MISTRAL_MODEL  # Маленькая модель для основной работы
        self.url = MISTRAL_URL
        self.client: httpx.AsyncClient | None = None
        
    async def init_client(self):
        """Создаём асинхронный клиент один раз при старте приложения"""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=60.0)
            
    async def close_client(self):
        """Закрываем клиент при завершении приложения"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    def build_prompt_extract(self, resume_text:str = None) -> dict:
        
        prompt_text = UC_ANALYSIS_PROMPT.format_messages(resume_text=resume_text)
       
        return "\n".join([m.content for m in prompt_text])
        
    def build_prompt_evaluation(self, resume_analysis: dict, vacancy_data: dict, 
                                criteria_weights: dict = None, candidate_id: str = None) -> dict:
        
        # Веса по умолчанию 
        default_weights = {
            'job_title_weight': 0.2,
            'education_weight': 0.15, 
            'experience_weight': 0.25,
            'schedule_weight': 0.05,
            'format_weight': 0.05,
            'additional_weight': 0.3
        }
        
        weights = {**default_weights, **(criteria_weights or {})}
        
        # Генерируем candidate_id если не передан
        if candidate_id is None:
            candidate_id = f"candidate_{int(time.time())}"
        
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
        
        return "\n".join([m.content for m in prompt_text])

    def build_prompt_questions(self, resume_analysis: dict, vacancy_requirements: dict,
                                    skill_gaps: list = None, strengths: list = None, 
                                    experience_summary: str = None) -> dict:
        
        prompt_text = UC_QUESTION_GENERATION_PROMPT.format_messages(
            resume_analysis=str(resume_analysis),
            vacancy_requirements=str(vacancy_requirements),
            additional_instructions="Сгенерируй разнообразные вопросы, охватывающие все аспекты кандидата",
            skill_gaps=", ".join(skill_gaps) if skill_gaps else "не выявлены",
            strengths=", ".join(strengths) if strengths else "опыт работы",
            experience_summary=experience_summary or "требует уточнения"
        )
        
        return "\n".join([m.content for m in prompt_text])
    
    def parsing_from_resumes(self, file_path: str = None) -> str:
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

        return cleaned_text

    @observe()
    async def extract_data_from_resume(self, file_path: str = None):
        
        """Извлечение данных из резюме"""

        cleaned_text = self.parsing_from_resumes(file_path)

        prompt_text = self.build_prompt_extract(resume_text=cleaned_text)

        try:
            # Анализируем резюме
            print("🔍 Анализируем резюме...")

            # Обновляем информацию о генерации
            langfuse.update_current_generation(
                name="extract_from_resumes",
                input={
                    "filepath": file_path
                },
                model=self.model,
                metadata={"task_type": "extract_from_resumes"}
            )
            
            # Вызываем Mistral API
            result = await self._call_mistral_api(prompt_text)
            
            # Подсчет токенов и стоимости
            output_text = str(result)
            usage_details, cost = count_tokens_and_cost(prompt_text, output_text)
            
            langfuse.update_current_generation(
                output=result,
                usage_details=usage_details,
                cost_details={"total": cost}
            )
                        
            if "error" not in result:
                print("✅ Анализ завершен успешно")
            else:
                print("❌ Ошибка анализа:", result["error"])
                
            return result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    @observe()
    async def evaluate_candidate_match(self, resume_analysis: dict, vacancy_data: dict, 
                                criteria_weights: dict = None, candidate_id: str = None) -> dict:
        """
        Оценка соответствия кандидата вакансии (только matching) - ТОЛЬКО АСИНХРОННЫЙ
        """

        prompt_text = self.build_prompt_evaluation(
            resume_analysis=resume_analysis, 
            vacancy_data=vacancy_data, 
            criteria_weights=criteria_weights, 
            candidate_id=candidate_id)

        try:
            print("🎯 Оцениваем соответствие кандидата вакансии...")
            
            # Обновляем информацию о генерации
            langfuse.update_current_generation(
                name="candidate_matching",
                input={
                    "resume_analysis": resume_analysis,
                    "vacancy_data": vacancy_data,
                    "criteria_weights": criteria_weights
                },
                model=self.model,
                metadata={"task_type": "candidate_matching"}
            )
            
            result = await self._call_mistral_api(prompt_text)
            
            # Подсчет токенов и стоимости
            output_text = str(result)
            usage_details, cost = count_tokens_and_cost(prompt_text, output_text)
            
            langfuse.update_current_generation(
                output=result,
                usage_details=usage_details,
                cost_details={"total": cost},
                metadata={"candidate_id": candidate_id}
            )
            
            overall_score = result.get('matching_results', {}).get('overall_score', 0)
            print(f"✅ Оценка соответствия: {overall_score}/100")
            
            # Обновляем trace с результатом
            langfuse.update_current_generation(
                output=result,
                metadata={"overall_score": overall_score}
            )
            
            return result
            
        except Exception as e:
            langfuse.update_current_generation(
                output={"error": str(e)},
                level="ERROR"
            )
            return {"error": f"Candidate evaluation failed: {str(e)}"}

    @observe()
    async def generate_interview_questions(self, resume_analysis: dict, vacancy_requirements: dict,
                                    skill_gaps: list = None, strengths: list = None, 
                                    experience_summary: str = None) -> dict:
        """
        Генерация вопросов для интервью - ТОЛЬКО АСИНХРОННЫЙ
        """

        prompt_text = self.build_prompt_questions(
            resume_analysis=resume_analysis, 
            vacancy_requirements=vacancy_requirements,
            skill_gaps=skill_gaps, 
            strengths=strengths, 
            experience_summary=experience_summary)

        try:
            print("📝 Генерируем вопросы для интервью...")
            
            # Обновляем информацию о генерации
            langfuse.update_current_generation(
                name="questions_generation",
                input={
                    "resume_analysis": resume_analysis,
                    "vacancy_requirements": vacancy_requirements,
                    "skill_gaps": skill_gaps,
                    "strengths": strengths
                },
                model=self.model,
                metadata={"task_type": "questions_generation"}
            )
            
            result = await self._call_mistral_api(prompt_text)
            
            # Подсчет токенов и стоимости
            output_text = str(result)
            usage_details, cost = count_tokens_and_cost(prompt_text, output_text)
            
            questions_count = len(result.get('questions', [])) if isinstance(result, dict) else 0
            
            langfuse.update_current_generation(
                output=result,
                usage_details=usage_details,
                cost_details={"total": cost},
                metadata={"questions_count": questions_count}
            )
            
            if "error" in result:
                langfuse.update_current_generation(
                    output={"error": result["error"]},
                    level="ERROR"
                )
                return {"error": f"Questions generation failed: {result['error']}"}

            return result
            
        except Exception as e:
            langfuse.update_current_generation(
                output={"error": str(e)},
                level="ERROR"
            )
            return {"error": f"Interview questions generation failed: {str(e)}"}
    
    async def _call_mistral_api(self, prompt_text: str) -> dict:
        """Вызов Mistral API - ОДНА асинхронная функция"""
        if self.client is None:
            await self.init_client()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"}
        }
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.post(self.url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    return self._parse_json_response(content)
                else:
                    if attempt == MAX_RETRIES:
                        return {"error": f"Mistral API error: {response.status_code}"}
                    await asyncio.sleep(RETRY_DELAY)
                    
            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"error": f"Mistral API call failed: {str(e)}"}
                await asyncio.sleep(RETRY_DELAY)
        
        return {"error": "Failed after retries"}
    
    def _parse_json_response(self, text: str) -> dict:
        """Парсинг JSON ответа от модели - синхронный, так как нет I/O"""
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
        
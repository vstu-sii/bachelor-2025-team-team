import os
import json
import time
import re
import requests
import glob
import asyncio
import random
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
import sys
from dataclasses import dataclass, field
from typing import Dict, List
import tiktoken  # для подсчета токенов

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import HRBaseline, langfuse
from ml.utils.file_parser import FileParser

# Загружаем ключи
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-large-latest"

# ============================================================================
# ПРОМПТЫ ДЛЯ A/B ТЕСТИРОВАНИЯ
# ============================================================================

# 1. ПРОМПТЫ ДЛЯ ИЗВЛЕЧЕНИЯ ДАННЫХ ИЗ РЕЗЮМЕ
EXTRACTION_PROMPTS = {
    "baseline": {
        "system": """Ты — HR-ассистент для анализа резюме.
Твоя задача: проанализировать резюме и извлечь структурированную информацию.

Правила анализа:
- Извлекай только факты, указанные в резюме
- Не добавляй информацию, которой нет в тексте
- Для навыков указывай только те, что явно указаны
- Для опыта считай только подтвержденные периоды работы
- Для образования указывай только указанные учреждения и степени

ВАЖНО: Верни ТОЛЬКО JSON без каких-либо комментариев, объяснений или дополнительного текста.

Формат вывода строго в JSON:
{{
  "contacts": {{
    "name": "имя кандидата",
    "sex": "пол кандидата", 
    "city": "город",
    "number": "телефон",
    "email": "почта",
    "social": ["список социальных сетей"]
  }}, 
  "skills": {{
    "technical": ["список технических навыков"],
    "soft": ["список мягких навыков"], 
    "languages": ["список языков"]
  }},
  "experience": {{
    "total_years": "число",
    "relevant_years": "число",
    "positions": [
      {{
        "title": "должность",
        "company": "название компании", 
        "sphere": "сектор, сфера", 
        "years": "число",
        "description": "описание обязанностей"
      }}
    ]
  }},
  "education": [
    {{
      "institution": "учебное заведение",
      "degree": "степень",
      "year": "год окончания",
      "field": "специальность"
    }}
  ]
}}""",
        "user": "Проанализируй это резюме и верни ТОЛЬКО JSON в указанном формате: {resume_text}"
    },
    
    "detailed": {
        "system": """Ты — опытный HR-аналитик с 10-летним стажем. 
Твоя задача: максимально подробно и точно извлечь ВСЕ данные из резюме.

ВНИМАНИЕ к деталям:
1. Контакты: Извлеки ВСЕ доступные контактные данные
2. Навыки: Раздели на технические, soft skills, инструменты, методологии
3. Опыт: Для каждой позиции укажи ТОЧНЫЕ даты, достижения, используемые технологии
4. Образование: Все степени, курсы, сертификаты, награды
5. Дополнительно: Проекты, публикации, волонтерство, хобби

Будь предельно точным. Если есть неясности — отмечай их в поле "notes".

ВАЖНО: Верни ТОЛЬКО JSON без каких-либо комментариев, объяснений или дополнительного текста.

Формат вывода строго в JSON:
{{
  "contacts": {{
    "name": "имя кандидата",
    "sex": "пол кандидата", 
    "city": "город",
    "number": "телефон",
    "email": "почта",
    "social": ["список социальных сетей"],
    "additional_contacts": ["дополнительные контакты если есть"]
  }}, 
  "skills": {{
    "technical": ["список технических навыков"],
    "soft": ["список мягких навыков"], 
    "languages": ["список языков"],
    "tools": ["инструменты"],
    "methodologies": ["методологии"]
  }},
  "experience": {{
    "total_years": "число",
    "relevant_years": "число",
    "positions": [
      {{
        "title": "должность",
        "company": "название компании", 
        "sphere": "сектор, сфера", 
        "start_date": "дата начала",
        "end_date": "дата окончания",
        "years": "число",
        "achievements": ["достижения"],
        "technologies": ["используемые технологии"],
        "description": "описание обязанностей"
      }}
    ]
  }},
  "education": [
    {{
      "institution": "учебное заведение",
      "degree": "степень",
      "year": "год окончания",
      "field": "специальность",
      "courses": ["пройденные курсы"],
      "certifications": ["сертификаты"],
      "awards": ["награды"]
    }}
  ],
  "additional": {{
    "projects": ["проекты"],
    "publications": ["публикации"],
    "volunteering": ["волонтерство"],
    "hobbies": ["хобби"]
  }},
  "notes": ["заметки о неясностях"]
}}""",
        "user": "Детально проанализируй это резюме и верни ТОЛЬКО JSON в указанном формате:\n\n{resume_text}"
    },
    
    "minimal": {
        "system": """Извлеки только КЛЮЧЕВЫЕ данные из резюме. 
Только самое важное: имя, ключевые навыки, последнее место работы, образование.
Без подробностей. Без интерпретаций.

ВАЖНО: Верни ТОЛЬКО JSON без каких-либо комментариев, объяснений или дополнительного текста.

Формат вывода строго в JSON:
{{
  "name": "имя кандидата",
  "key_skills": ["ключевые навыки"],
  "last_position": {{
    "title": "должность",
    "company": "компания",
    "years": "стаж в годах"
  }},
  "education": {{
    "highest_degree": "высшая степень",
    "institution": "учебное заведение"
  }},
  "summary": "краткое резюме в 1-2 предложениях"
}}""",
        "user": "Извлеки ключевые данные из резюме и верни ТОЛЬКО JSON в указанном формате:\n\nРезюме: {resume_text}"
    },
    
    "structured": {
        "system": """Ты специалист по структурированию данных. 
Преобразуй неструктурированное резюме в четко организованные данные.

Требования:
1. Стандартизируй названия должностей (приводи к common titles)
2. Группируй похожие навыки (убирай дубликаты)
3. Приводи даты к единому формату (ГГГГ-ММ)
4. Классифицируй компании по отраслям (IT, Finance, Retail и т.д.)
5. Определяй уровень seniority (Junior, Middle, Senior, Lead)

Цель: сделать данные максимально удобными для автоматической обработки.

ВАЖНО: Верни ТОЛЬКО JSON без каких-либо комментариев, объяснений или дополнительного текста.

Формат вывода строго в JSON:
{{
  "standardized_data": {{
    "contacts": {{
      "name": "стандартизированное имя",
      "city": "стандартизированный город"
    }},
    "skills": {{
      "technical": ["стандартизированные технические навыки"],
      "soft": ["стандартизированные мягкие навыки"]
    }},
    "experience": {{
      "total_years": "число",
      "seniority_level": "Junior/Middle/Senior/Lead",
      "positions": [
        {{
          "title": "стандартизированная должность",
          "company": "компания",
          "industry": "отрасль",
          "start_date": "ГГГГ-ММ",
          "end_date": "ГГГГ-ММ или 'настоящее время'",
          "duration_months": "число"
        }}
      ]
    }},
    "education": [
      {{
        "institution": "учебное заведение",
        "degree_level": "Bachelor/Master/PhD/Certificate",
        "graduation_year": "ГГГГ"
      }}
    ]
  }},
  "metadata": {{
    "processing_date": "ГГГГ-ММ-ДД",
    "data_quality_score": 0-100
  }}
}}""",
        "user": "Структурируй данные из резюме и верни ТОЛЬКО JSON в указанном формате:\n{resume_text}"
    }
}

# 2. ПРОМПТЫ ДЛЯ ОЦЕНКИ СООТВЕТСТВИЯ ВАКАНСИИ
EVALUATION_PROMPTS = {
    "baseline": {
        "system": """Ты — HR-эксперт по подбору персонала. 
Оцени соответствие кандидата вакансии по 6 критериям с весами.

ВАЖНО:
1. Верни ТОЛЬКО JSON без каких-либо комментариев
2. Ограничь текстовые поля до 200 символов
3. Используй только числа для оценок

JSON структура:
{
  "scores": {
    "job_title": {"score": 0-100, "weight": 0.2, "explanation": "до 100 символов"},
    "education": {"score": 0-100, "weight": 0.15, "explanation": "до 100 символов"},
    "experience": {"score": 0-100, "weight": 0.25, "explanation": "до 100 символов"},
    "schedule": {"score": 0-100, "weight": 0.05, "explanation": "до 100 символов"},
    "format": {"score": 0-100, "weight": 0.05, "explanation": "до 100 символов"},
    "additional": {"score": 0-100, "weight": 0.3, "explanation": "до 100 символов"}
  },
  "overall": {
    "score": 0-100,
    "suitable": true/false,
    "recommendation": "рекомендован/условно/нет (до 50 символов)"
  }
}""",
        "user": """Вакансия: {vacancy_data}
Кандидат: {resume_analysis}
Оцени и верни JSON."""
    },
    
    "strategic": {
        "system": """Ты стратегический HR-консультант. Дай краткую оценку.

ВАЖНО:
1. Верни ТОЛЬКО JSON
2. Все текстовые поля до 150 символов
3. Оценки только числа

JSON:
{
  "current_fit": {
    "job_fit": 0-100,
    "skills_fit": 0-100,
    "experience_fit": 0-100
  },
  "strategic": {
    "growth_potential": 0-100,
    "cultural_fit": 0-100,
    "risk_level": "низкий/средний/высокий"
  },
  "recommendation": {
    "decision": "рекомендую/условно/нет",
    "priority": "высокий/средний/низкий"
  }
}""",
        "user": """Вакансия: {vacancy_data}
Кандидат: {resume_analysis}
Стратегическая оценка. Только JSON."""
    },
    
    "technical": {
        "system": """Ты технический рекрутер. Оцени техническое соответствие.

ВАЖНО: ТОЛЬКО JSON, тексты до 100 символов.

JSON:
{
  "technical_match": 0-100,
  "missing_critical": ["технология1", "технология2"],
  "strengths": ["сила1", "сила2"],
  "interview_focus": ["тема1", "тема2"]
}""",
        "user": """Требования: {vacancy_data}
Навыки: {resume_analysis}
Техническая оценка. Только JSON."""
    }
}

# 3. ПРОМПТЫ ДЛЯ ГЕНЕРАЦИИ ВОПРОСОВ
QUESTION_PROMPTS = {
    "baseline": {
        "system": """Ты — эксперт по проведению интервью.
Сгенерируй 5-8 вопросов для собеседования.

ВАЖНО:
1. ТОЛЬКО JSON
2. Каждый вопрос до 150 символов
3. Цель вопроса до 50 символов
4. Максимальная простота структуры

JSON:
{
  "questions": [
    {
      "text": "текст вопроса",
      "type": "технический/поведенческий/кейсовый",
      "purpose": "цель",
      "time": "1-3 минуты"
    }
  ],
  "total": "число"
}""",
        "user": """Резюме: {resume_analysis}
Вакансия: {vacancy_requirements}
Сгенерируй 5-8 вопросов. Только JSON."""
    },
    
    "behavioral": {
        "system": """Сгенерируй 4-6 поведенческих вопросов по STAR.

ВАЖНО: ТОЛЬКО JSON, вопросы до 100 символов.

JSON:
{
  "star_questions": [
    {
      "question": "вопрос",
      "competency": "компетенция",
      "focus": "ситуация/задача/действие/результат"
    }
  ]
}""",
        "user": """Кандидат: {resume_analysis}
4-6 поведенческих вопросов. Только JSON."""
    },
    
    "deep_dive": {
        "system": """Сгенерируй 3-5 углубленных вопросов для senior.

ВАЖНО: ТОЛЬКО JSON, вопросы до 120 символов.

JSON:
{
  "deep_questions": [
    {
      "question": "вопрос",
      "area": "архитектура/лидерство/стратегия",
      "level": "senior/lead"
    }
  ]
}""",
        "user": """Senior кандидат: {resume_analysis}
Вакансия: {vacancy_requirements}
3-5 углубленных вопросов. Только JSON."""
    }
}

# ============================================================================
# СТАНДАРТНЫЕ ПРОМПТЫ ДЛЯ ОЦЕНКИ КАЧЕСТВА (ЕДИНЫЕ ДЛЯ ВСЕХ ТЕСТОВ)
# ============================================================================

QUALITY_EVALUATION_PROMPTS = {
    "extraction": """Ты проверяющий ассистент. Оцени качество извлечения данных из резюме.

Оригинальное резюме:
{resume_text}

Извлеченные данные:
{extracted_data}

Критерии оценки (0-100):
1. ПОЛНОТА: Все ли данные из резюме извлечены
2. ТОЧНОСТЬ: Точно ли воспроизведена информация
3. СТРУКТУРА: Логично ли организованы данные
4. ДЕТАЛЬНОСТЬ: Не пропущены ли важные детали

Верни JSON:
{{
  "completeness_score": 0-100,
  "accuracy_score": 0-100,
  "structure_score": 0-100,
  "detail_score": 0-100,
  "overall_score": 0-100,
  "missing_data": ["список пропущенных данных"],
  "incorrect_data": ["список ошибок"],
  "suggestions": ["рекомендации по улучшению"]
}}""",
    
    "evaluation": """Ты HR-эксперт. Оцени качество оценки соответствия кандидата вакансии.

Данные кандидата:
{resume_data}

Требования вакансии:
{vacancy_data}

Оценка соответствия:
{evaluation_result}

Критерии (0-100):
1. ОБОСНОВАННОСТЬ: Логичны ли аргументы в оценке
2. БАЛАНСИРОВАННОСТЬ: Учтены ли все аспекты
3. ПРАКТИЧЕСКАЯ ПОЛЕЗНОСТЬ: Поможет ли оценка принять решение
4. ДЕТАЛЬНОСТЬ: Достаточно ли подробный анализ

JSON:
{{
  "reasoning_score": 0-100,
  "balance_score": 0-100,
  "practicality_score": 0-100,
  "detail_score": 0-100,
  "overall_score": 0-100,
  "strengths": ["сильные стороны оценки"],
  "weaknesses": ["слабые стороны"],
  "improvements": ["предложения по улучшению"]
}}""",
    
    "questions": """Ты senior HR. Оцени качество вопросов для интервью.

Данные кандидата:
{resume_data}

Вакансия:
{vacancy_data}

Сгенерированные вопросы:
{generated_questions}

Критерии (0-100):
1. РЕЛЕВАНТНОСТЬ: Соответствуют ли вопросы резюме и вакансии
2. РАЗНООБРАЗИЕ: Разные типы вопросов, разные темы
3. ГЛУБИНА: Насколько глубоко раскрывают компетенции
4. ПРАКТИЧЕСКАЯ ЦЕННОСТЬ: Помогут ли оценить кандидата

JSON:
{{
  "relevance_score": 0-100,
  "diversity_score": 0-100,
  "depth_score": 0-100,
  "practical_value_score": 0-100,
  "overall_score": 0-100,
  "best_questions": ["3 лучших вопроса"],
  "weak_questions": ["3 слабых вопроса"],
  "recommendations": ["рекомендации"]
}}"""
}

# ============================================================================
# МЕНЕДЖЕР A/B ТЕСТИРОВАНИЯ
# ============================================================================

@dataclass
class TestResult:
    """Результат одного теста"""
    test_type: str  # extraction, evaluation, questions
    variant: str    # название варианта промпта
    prompt_used: Dict[str, str]  # использованный промпт
    result_data: Any  # результат выполнения
    quality_metrics: Dict[str, float]  # метрики качества
    performance: Dict[str, float]  # производительность
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ABTestManager:
    """Управление A/B тестированием промптов"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.best_variants = {}
        
    def add_result(self, result: TestResult):
        """Добавить результат теста"""
        self.results.append(result)
        
    def get_best_variant(self, test_type: str) -> Optional[Tuple[str, Dict]]:
        """Получить лучший вариант для типа теста"""
        type_results = [r for r in self.results if r.test_type == test_type]
        if not type_results:
            return None
            
        # Группируем по вариантам
        variants = {}
        for result in type_results:
            if result.variant not in variants:
                variants[result.variant] = []
            variants[result.variant].append(result)
        
        # Находим вариант с наивысшим средним overall_score
        best_variant = None
        best_score = -1
        
        for variant, results in variants.items():
            avg_score = sum(r.quality_metrics.get('overall_score', 0) for r in results) / len(results)
            if avg_score > best_score:
                best_score = avg_score
                best_variant = variant
        
        if best_variant:
            # Собираем статистику по лучшему варианту
            best_results = variants[best_variant]
            stats = {
                "avg_overall_score": best_score,
                "avg_latency": sum(r.performance.get('latency', 0) for r in best_results) / len(best_results),
                "total_tests": len(best_results),
                "sample_prompt": best_results[0].prompt_used,
                "sample_result": best_results[0].result_data
            }
            
            self.best_variants[test_type] = (best_variant, stats)
            return best_variant, stats
        
        return None
    
    def generate_report(self) -> Dict:
        """Сгенерировать полный отчет"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "by_test_type": {},
            "best_variants": {},
            "summary": {}
        }
        
        # Группируем по типам тестов
        for test_type in ["extraction", "evaluation", "questions"]:
            type_results = [r for r in self.results if r.test_type == test_type]
            if not type_results:
                continue
                
            report["by_test_type"][test_type] = {
                "total": len(type_results),
                "variants_tested": list(set(r.variant for r in type_results)),
                "average_scores": self._calculate_average_scores(type_results)
            }
        
        # Добавляем лучшие варианты
        for test_type in ["extraction", "evaluation", "questions"]:
            best = self.get_best_variant(test_type)
            if best:
                variant, stats = best
                report["best_variants"][test_type] = {
                    "variant": variant,
                    "avg_score": stats["avg_overall_score"],
                    "avg_latency": stats["avg_latency"],
                    "total_tests": stats["total_tests"]
                }
        
        # Сводка
        report["summary"] = {
            "overall_best": max(
                [(k, v["avg_score"]) for k, v in report["best_variants"].items()],
                key=lambda x: x[1]
            ) if report["best_variants"] else None,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _calculate_average_scores(self, results: List[TestResult]) -> Dict:
        """Рассчитать средние метрики"""
        if not results:
            return {}
            
        metrics = {}
        count = len(results)
        
        for result in results:
            for key, value in result.quality_metrics.items():
                if key not in metrics:
                    metrics[key] = 0
                metrics[key] += value
        
        return {k: v/count for k, v in metrics.items()}
    
    def _generate_recommendations(self) -> List[str]:
        """Сгенерировать рекомендации"""
        recs = []
        
        for test_type in ["extraction", "evaluation", "questions"]:
            best = self.get_best_variant(test_type)
            if best:
                variant, stats = best
                recs.append(f"{test_type.upper()}: Используйте вариант '{variant}' (средняя оценка: {stats['avg_overall_score']:.1f}/100)")
        
        if not recs:
            recs.append("Недостаточно данных для рекомендаций. Запустите больше тестов.")
        
        return recs
    
    def save_report(self, filename: str = None):
        """Сохранить отчет в файл"""
        if filename is None:
            filename = f"ab_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Отчет сохранен: {filename}")
        
        # Также сохраняем текстовую версию
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(self._format_text_report(report))
        
        return report
    
    def _format_text_report(self, report: Dict) -> str:
        """Форматировать текстовый отчет"""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 ОТЧЕТ ПО A/B ТЕСТИРОВАНИЮ ПРОМПТОВ")
        lines.append(f"📅 Сгенерирован: {report['generated_at']}")
        lines.append(f"🧪 Всего тестов: {report['total_tests']}")
        lines.append("=" * 60)
        
        for test_type, data in report.get('by_test_type', {}).items():
            lines.append(f"\n🔧 {test_type.upper()}:")
            lines.append(f"   Тестов: {data['total']}")
            lines.append(f"   Вариантов: {', '.join(data['variants_tested'])}")
            
            for metric, value in data.get('average_scores', {}).items():
                lines.append(f"   {metric}: {value:.1f}/100")
        
        lines.append("\n" + "=" * 60)
        lines.append("🏆 ЛУЧШИЕ ВАРИАНТЫ:")
        
        for test_type, best in report.get('best_variants', {}).items():
            lines.append(f"\n   {test_type.upper()}:")
            lines.append(f"     Вариант: {best['variant']}")
            lines.append(f"     Средняя оценка: {best['avg_score']:.1f}/100")
            lines.append(f"     Среднее время: {best['avg_latency']:.2f} сек")
        
        lines.append("\n" + "=" * 60)
        lines.append("💡 РЕКОМЕНДАЦИИ:")
        
        for rec in report.get('summary', {}).get('recommendations', []):
            lines.append(f"   • {rec}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)

# ============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С LLM
# ============================================================================

def clean_llm_output(output: str) -> str:
    if not isinstance(output, str):
        return ""

    text = output.strip()

    # Убираем markdown
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # Ищем первый { и последнюю }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    json_text = text[start:end + 1]

    # Убираем управляющие символы
    json_text = re.sub(r'[\x00-\x1f\x7f]', '', json_text)

    return json_text.strip()

def call_mistral_api(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> Dict:
    """Вызов API Mistral"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    
    try:
        start_time = time.time()
        resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start_time
        
        if resp.status_code != 200:
            return {"error": f"API error: {resp.status_code}", "latency": latency}
        
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not content:
            return {"error": "Empty response", "latency": latency}
        
        cleaned = clean_llm_output(content)
        
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}", "raw_output": content, "latency": latency}
        
        return {
            "result": parsed,
            "latency": latency,
            "usage": data.get("usage", {}),
            "raw_content": content
        }
        
    except Exception as e:
        return {"error": f"API call failed: {e}", "latency": latency if 'latency' in locals() else 0}

def evaluate_quality(test_type: str, data_for_evaluation: Dict) -> Dict:
    """Оценка качества результата (единый для всех вариантов)"""
    
    if test_type not in QUALITY_EVALUATION_PROMPTS:
        return {"error": f"Unknown test type: {test_type}"}
    
    prompt = QUALITY_EVALUATION_PROMPTS[test_type].format(**data_for_evaluation)
    
    result = call_mistral_api(
        system_prompt="Ты опытный оценщик качества. Будь объективным и строгим.",
        user_prompt=prompt,
        max_tokens=1024
    )
    
    if "error" in result:
        return result
    
    # Добавляем базовые метрики если их нет
    if "overall_score" not in result["result"]:
        scores = [v for k, v in result["result"].items() if "score" in k.lower() and isinstance(v, (int, float))]
        if scores:
            result["result"]["overall_score"] = sum(scores) / len(scores)
    
    return result["result"]

# ============================================================================
# ФУНКЦИИ A/B ТЕСТИРОВАНИЯ
# ============================================================================

async def run_extraction_ab_test(resume_text: str, variants_to_test: List[str] = None) -> List[TestResult]:
    """
    A/B тест промптов для извлечения данных из резюме
    
    Каждый вариант промпта выполняется отдельно, 
    результаты оцениваются единым стандартным способом
    """
    
    if variants_to_test is None:
        variants_to_test = list(EXTRACTION_PROMPTS.keys())
    
    print(f"\n🧪 A/B ТЕСТ ИЗВЛЕЧЕНИЯ ДАННЫХ")
    print(f"Тестируем варианты: {', '.join(variants_to_test)}")
    
    results = []
    
    for variant in variants_to_test:
        print(f"\n  🎯 Вариант: {variant}")
        
        if variant not in EXTRACTION_PROMPTS:
            print(f"    ❌ Вариант не найден: {variant}")
            continue
        
        prompt_config = EXTRACTION_PROMPTS[variant]
        
        # Шаг 1: Извлекаем данные с ЭТИМ промптом
        extraction_result = call_mistral_api(
            system_prompt=prompt_config["system"],
            user_prompt=prompt_config["user"].format(resume_text=resume_text)
        )
        
        if "error" in extraction_result:
            print(f"    ❌ Ошибка извлечения: {extraction_result['error']}")
            continue
        
        # Шаг 2: Оцениваем качество извлечения ЕДИНЫМ способом
        quality_data = {
            "resume_text": resume_text[:5000],  # Ограничиваем для экономии токенов
            "extracted_data": json.dumps(extraction_result["result"], ensure_ascii=False)
        }
        
        quality_evaluation = evaluate_quality("extraction", quality_data)
        
        if "error" in quality_evaluation:
            print(f"    ❌ Ошибка оценки: {quality_evaluation['error']}")
            continue
        
        # Шаг 3: Сохраняем результат
        test_result = TestResult(
            test_type="extraction",
            variant=variant,
            prompt_used=prompt_config,
            result_data=extraction_result["result"],
            quality_metrics=quality_evaluation,
            performance={
                "latency": extraction_result["latency"],
                "total_tokens": extraction_result.get("usage", {}).get("total_tokens", 0)
            }
        )
        
        results.append(test_result)
        
        print(f"    ✅ Качество извлечения: {quality_evaluation.get('overall_score', 0):.1f}/100")
        print(f"    ⏱️  Время выполнения: {extraction_result['latency']:.2f} сек")
        print(f"    📊 Детали: ", end="")
        for key, value in quality_evaluation.items():
            if "score" in key and isinstance(value, (int, float)):
                print(f"{key}: {value:.1f} ", end="")
        print()
    
    return results

async def run_evaluation_ab_test(resume_data: Dict, vacancy_data: Dict, variants_to_test: List[str] = None) -> List[TestResult]:
    """
    A/B тест промптов для оценки соответствия вакансии
    """
    
    if variants_to_test is None:
        variants_to_test = list(EVALUATION_PROMPTS.keys())
    
    print(f"\n🧪 A/B ТЕСТ ОЦЕНКИ СООТВЕТСТВИЯ")
    print(f"Тестируем варианты: {', '.join(variants_to_test)}")
    
    results = []
    
    for variant in variants_to_test:
        print(f"\n  🎯 Вариант: {variant}")
        
        if variant not in EVALUATION_PROMPTS:
            print(f"    ❌ Вариант не найден: {variant}")
            continue
        
        prompt_config = EVALUATION_PROMPTS[variant]
        
        # Шаг 1: Оцениваем соответствие с ЭТИМ промптом
        evaluation_result = call_mistral_api(
            system_prompt=prompt_config["system"],
            user_prompt=prompt_config["user"].format(
                resume_analysis=json.dumps(resume_data, ensure_ascii=False),
                vacancy_data=json.dumps(vacancy_data, ensure_ascii=False)
            )
        )
        
        if "error" in evaluation_result:
            print(f"    ❌ Ошибка оценки: {evaluation_result['error']}")
            continue
        
        # Шаг 2: Оцениваем качество оценки ЕДИНЫМ способом
        quality_data = {
            "resume_data": json.dumps(resume_data, ensure_ascii=False),
            "vacancy_data": json.dumps(vacancy_data, ensure_ascii=False),
            "evaluation_result": json.dumps(evaluation_result["result"], ensure_ascii=False)
        }
        
        quality_evaluation = evaluate_quality("evaluation", quality_data)
        
        if "error" in quality_evaluation:
            print(f"    ❌ Ошибка оценки качества: {quality_evaluation['error']}")
            continue
        
        # Шаг 3: Сохраняем результат
        test_result = TestResult(
            test_type="evaluation",
            variant=variant,
            prompt_used=prompt_config,
            result_data=evaluation_result["result"],
            quality_metrics=quality_evaluation,
            performance={
                "latency": evaluation_result["latency"],
                "total_tokens": evaluation_result.get("usage", {}).get("total_tokens", 0)
            }
        )
        
        results.append(test_result)
        
        print(f"    ✅ Качество оценки: {quality_evaluation.get('overall_score', 0):.1f}/100")
        print(f"    ⏱️  Время выполнения: {evaluation_result['latency']:.2f} сек")
    
    return results

async def run_questions_ab_test(resume_data: Dict, vacancy_data: Dict, variants_to_test: List[str] = None) -> List[TestResult]:
    """
    A/B тест промптов для генерации вопросов
    """
    
    if variants_to_test is None:
        variants_to_test = list(QUESTION_PROMPTS.keys())
    
    print(f"\n🧪 A/B ТЕСТ ГЕНЕРАЦИИ ВОПРОСОВ")
    print(f"Тестируем варианты: {', '.join(variants_to_test)}")
    
    results = []
    
    for variant in variants_to_test:
        print(f"\n  🎯 Вариант: {variant}")
        
        if variant not in QUESTION_PROMPTS:
            print(f"    ❌ Вариант не найден: {variant}")
            continue
        
        prompt_config = QUESTION_PROMPTS[variant]
        
        # Шаг 1: Генерируем вопросы с ЭТИМ промптом
        questions_result = call_mistral_api(
            system_prompt=prompt_config["system"],
            user_prompt=prompt_config["user"].format(
                resume_analysis=json.dumps(resume_data, ensure_ascii=False),
                vacancy_requirements=json.dumps(vacancy_data, ensure_ascii=False)
            )
        )
        
        if "error" in questions_result:
            print(f"    ❌ Ошибка генерации: {questions_result['error']}")
            continue
        
        # Шаг 2: Оцениваем качество вопросов ЕДИНЫМ способом
        quality_data = {
            "resume_data": json.dumps(resume_data, ensure_ascii=False),
            "vacancy_data": json.dumps(vacancy_data, ensure_ascii=False),
            "generated_questions": json.dumps(questions_result["result"], ensure_ascii=False)
        }
        
        quality_evaluation = evaluate_quality("questions", quality_data)
        
        if "error" in quality_evaluation:
            print(f"    ❌ Ошибка оценки качества: {quality_evaluation['error']}")
            continue
        
        # Шаг 3: Сохраняем результат
        test_result = TestResult(
            test_type="questions",
            variant=variant,
            prompt_used=prompt_config,
            result_data=questions_result["result"],
            quality_metrics=quality_evaluation,
            performance={
                "latency": questions_result["latency"],
                "total_tokens": questions_result.get("usage", {}).get("total_tokens", 0)
            }
        )
        
        results.append(test_result)
        
        print(f"    ✅ Качество вопросов: {quality_evaluation.get('overall_score', 0):.1f}/100")
        print(f"    ⏱️  Время выполнения: {questions_result['latency']:.2f} сек")
        
        # Показываем лучшие вопросы если есть
        if "best_questions" in quality_evaluation and quality_evaluation["best_questions"]:
            print(f"    🏆 Лучшие вопросы:")
            for i, q in enumerate(quality_evaluation["best_questions"][:2], 1):
                print(f"      {i}. {q[:80]}...")
    
    return results

# ============================================================================
# ИНТЕГРАЦИЯ С ВАШЕЙ СИСТЕМОЙ
# ============================================================================

def _find_test_files(test_folders):
    """Ищет тестовые файлы в различных папках"""
    test_files = []
    for folder in test_folders:
        if os.path.exists(folder):
            patterns = [f"{folder}/*.pdf", f"{folder}/*.docx", f"{folder}/*.doc", f"{folder}/*.txt"]
            for pattern in patterns:
                found_files = glob.glob(pattern)
                test_files.extend(found_files)
    return test_files

async def run_comprehensive_ab_test_on_files(test_count: int = 2):
    """
    Комплексный A/B тест на нескольких тестовых файлах
    """
    
    # Инициализируем менеджер
    ab_manager = ABTestManager()
    
    # Находим тестовые файлы
    test_files = _find_test_files(["C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/test_resumes/"])
    if not test_files:
        print("❌ Тестовые файлы не найдены")
        return ab_manager
    
    # Загружаем вакансию
    vacancy_path = "C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/vacancy.json"
    try:
        with open(vacancy_path, 'r', encoding='utf-8') as f:
            vacancy_data = json.load(f)
        print(f"✅ Загружена вакансия: {vacancy_data.get('position', 'Не указана')}")
    except Exception as e:
        print(f"❌ Ошибка загрузки вакансии: {e}")
        return ab_manager
    
    # Выбираем файлы для теста
    selected_files = test_files[:min(test_count, len(test_files))]
    
    print("\n" + "=" * 60)
    print(f"🚀 ЗАПУСК КОМПЛЕКСНОГО A/B ТЕСТИРОВАНИЯ")
    print(f"📁 Тестируем {len(selected_files)} файлов")
    print("=" * 60)
    
    for i, test_file in enumerate(selected_files, 1):
        print(f"\n📄 ФАЙЛ {i}/{len(selected_files)}: {os.path.basename(test_file)}")
        
        try:
            # Загружаем текст резюме (упрощенно)
            resume_text = parsing_from_resumes(test_file)
            
            # 1. Тестируем извлечение данных
            print("\n1. 🔧 Тестируем извлечение данных...")
            extraction_results = await run_extraction_ab_test(resume_text)
            
            for result in extraction_results:
                ab_manager.add_result(result)
            
            # Выбираем лучшее извлечение для дальнейших тестов
            if extraction_results:
                best_extraction = max(extraction_results, key=lambda x: x.quality_metrics.get('overall_score', 0))
                extracted_data = best_extraction.result_data
                
                # 2. Тестируем оценку соответствия
                print("\n2. 📊 Тестируем оценку соответствия...")
                evaluation_results = await run_evaluation_ab_test(extracted_data, vacancy_data)
                
                for result in evaluation_results:
                    ab_manager.add_result(result)
                
                # 3. Тестируем генерацию вопросов
                print("\n3. ❓ Тестируем генерацию вопросов...")
                questions_results = await run_questions_ab_test(extracted_data, vacancy_data)
                
                for result in questions_results:
                    ab_manager.add_result(result)
            
        except Exception as e:
            print(f"❌ Ошибка при обработке файла {test_file}: {e}")
            continue
    
    # Генерируем отчет
    print("\n" + "=" * 60)
    print("📈 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    report = ab_manager.save_report()
    
    # Выводим краткие результаты
    print("\n🏆 ИТОГИ:")
    for test_type in ["extraction", "evaluation", "questions"]:
        best = ab_manager.get_best_variant(test_type)
        if best:
            variant, stats = best
            print(f"  {test_type.upper()}: {variant} ({stats['avg_overall_score']:.1f}/100)")
    
    return ab_manager

def parsing_from_resumes(file_path: str = None) -> str:
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

async def quick_ab_test_demo():
    """Быстрая демонстрация A/B тестирования на одном файле"""
    
    # Находим тестовый файл
    test_file = "C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/test_resumes/1.docx"
    
    print(f"📁 Тестовый файл: {os.path.basename(test_file)}")
    
    # Загружаем вакансию
    vacancy_path = "C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/vacancy.json"
    try:
        with open(vacancy_path, 'r', encoding='utf-8') as f:
            vacancy_data = json.load(f)
    except:
        vacancy_data = {"position": "Тестовая вакансия", "requirements": "Требования"}
    
    # Загружаем резюме
    resume_text = parsing_from_resumes(test_file)
    
    # Инициализируем менеджер
    ab_manager = ABTestManager()
    
    # Тестируем только извлечение (для демо)
    print("\n🧪 ДЕМО: A/B тест извлечения данных")
    results = await run_extraction_ab_test(resume_text, ["baseline", "detailed", "minimal"])
    
    for result in results:
        ab_manager.add_result(result)
    
    # Показываем результаты
    best = ab_manager.get_best_variant("extraction")
    if best:
        variant, stats = best
        print(f"\n🏆 ЛУЧШИЙ ВАРИАНТ: {variant}")
        print(f"📊 Средняя оценка: {stats['avg_overall_score']:.1f}/100")
        print(f"⏱️  Среднее время: {stats['avg_latency']:.2f} сек")
    
    return ab_manager

async def interactive_ab_test():
    """Интерактивный режим A/B тестирования"""
    
    print("\n🎯 ИНТЕРАКТИВНОЕ A/B ТЕСТИРОВАНИЕ ПРОМПТОВ")
    print("=" * 50)
    
    ab_manager = ABTestManager()
    
    while True:
        print("\nЧто хотите протестировать?")
        print("1. Извлечение данных из резюме")
        print("2. Оценку соответствия вакансии")
        print("3. Генерацию вопросов для интервью")
        print("4. Комплексный тест всего")
        print("5. Показать текущие результаты")
        print("6. Сохранить отчет и выйти")
        print("=" * 30)
        
        choice = input("Выберите (1-6): ").strip()
        
        if choice == "1":
            # Тест извлечения
            test_file = input("Путь к файлу резюме (или Enter для тестового): ").strip()
            if not test_file:
                test_files = _find_test_files(["C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/test_resumes/"])
                if test_files:
                    test_file = test_files[0]
                else:
                    print("❌ Тестовые файлы не найдены")
                    continue
            
            resume_text = parsing_from_resumes(test_file)
            
            variants = input("Варианты для теста (через запятую или Enter для всех): ").strip()
            if variants:
                variants_to_test = [v.strip() for v in variants.split(",")]
            else:
                variants_to_test = None
            
            results = await run_extraction_ab_test(resume_text, variants_to_test)
            for result in results:
                ab_manager.add_result(result)
        
        elif choice == "2":
            # Тест оценки
            print("Для оценки нужны данные резюме и вакансии")
            
            # Загружаем тестовые данные
            test_files = _find_test_files(["C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/test_resumes/"])
            if not test_files:
                print("❌ Тестовые файлы не найдены")
                continue
            
            vacancy_path = "C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/vacancy.json"
            try:
                with open(vacancy_path, 'r', encoding='utf-8') as f:
                    vacancy_data = json.load(f)
            except:
                print("❌ Ошибка загрузки вакансии")
                continue
            
            # Сначала нужно извлечь данные (используем baseline)
            resume_text = parsing_from_resumes(test_files[0])
            
            print("📋 Извлекаем данные (baseline промпт)...")
            extraction_result = call_mistral_api(
                system_prompt=EXTRACTION_PROMPTS["baseline"]["system"],
                user_prompt=EXTRACTION_PROMPTS["baseline"]["user"].format(resume_text=resume_text)
            )
            
            if "error" in extraction_result:
                print(f"❌ Ошибка извлечения: {extraction_result['error']}")
                continue
            
            resume_data = extraction_result["result"]
            
            variants = input("Варианты для теста (через запятую или Enter для всех): ").strip()
            if variants:
                variants_to_test = [v.strip() for v in variants.split(",")]
            else:
                variants_to_test = None
            
            results = await run_evaluation_ab_test(resume_data, vacancy_data, variants_to_test)
            for result in results:
                ab_manager.add_result(result)
        
        elif choice == "3":
            # Тест вопросов
            print("Для генерации вопросов нужны данные резюме и вакансии")
            
            # Загружаем тестовые данные
            test_files = _find_test_files(["C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/test_resumes/"])
            if not test_files:
                print("❌ Тестовые файлы не найдены")
                continue
            
            vacancy_path = "C:/Users/67181/OneDrive/Dokumenty/bachelor-2025-team-team/data/vacancy.json"
            try:
                with open(vacancy_path, 'r', encoding='utf-8') as f:
                    vacancy_data = json.load(f)
            except:
                print("❌ Ошибка загрузки вакансии")
                continue
            
            # Извлекаем данные
            resume_text = parsing_from_resumes(test_files[0])
            
            print("📋 Извлекаем данные (baseline промпт)...")
            extraction_result = call_mistral_api(
                system_prompt=EXTRACTION_PROMPTS["baseline"]["system"],
                user_prompt=EXTRACTION_PROMPTS["baseline"]["user"].format(resume_text=resume_text)
            )
            
            if "error" in extraction_result:
                print(f"❌ Ошибка извлечения: {extraction_result['error']}")
                continue
            
            resume_data = extraction_result["result"]
            
            variants = input("Варианты для теста (через запятую или Enter для всех): ").strip()
            if variants:
                variants_to_test = [v.strip() for v in variants.split(",")]
            else:
                variants_to_test = None
            
            results = await run_questions_ab_test(resume_data, vacancy_data, variants_to_test)
            for result in results:
                ab_manager.add_result(result)
        
        elif choice == "4":
            # Комплексный тест
            count = input("Количество файлов для теста (по умолчанию 2): ").strip()
            test_count = int(count) if count.isdigit() else 2
            
            await run_comprehensive_ab_test_on_files(test_count)
        
        elif choice == "5":
            # Показать результаты
            if not ab_manager.results:
                print("📭 Нет результатов тестов")
            else:
                report = ab_manager.generate_report()
                print(ab_manager._format_text_report(report))
        
        elif choice == "6":
            # Сохранить и выйти
            if ab_manager.results:
                ab_manager.save_report()
            print("👋 Завершение работы")
            break
        
        else:
            print("❌ Неверный выбор")

# ============================================================================
# МЭЙН ФУНКЦИИ
# ============================================================================

async def main():
    """Главная функция"""
    
    print("\n" + "=" * 60)
    print("🤖 СИСТЕМА A/B ТЕСТИРОВАНИЯ ПРОМПТОВ")
    print("=" * 60)
    print("Варианты запуска:")
    print("1. Комплексный тест на нескольких файлах")
    print("2. Быстрая демонстрация на одном файле")
    print("3. Интерактивный режим")
    print("=" * 60)
    
    choice = input("Выберите вариант (1-3): ").strip()
    
    if choice == "1":
        count = input("Количество файлов (по умолчанию 2): ").strip()
        test_count = int(count) if count.isdigit() else 2
        await run_comprehensive_ab_test_on_files(test_count)
    
    elif choice == "2":
        await quick_ab_test_demo()
    
    elif choice == "3":
        await interactive_ab_test()
    
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    asyncio.run(main())
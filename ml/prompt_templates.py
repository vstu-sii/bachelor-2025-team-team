from langchain_core.prompts import ChatPromptTemplate
import sys
import os

UC_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — HR-ассистент для анализа резюме.
Твоя задача: проанализировать резюме и извлечь структурированную информацию.

Правила анализа:
- Извлекай только факты, указанные в резюме
- Не добавляй информацию, которой нет в тексте
- Для навыков указывай только те, что явно указаны
- Для опыта считай только подтвержденные периоды работы
- Для образования указывай только указанные учреждения и степени

Формат вывода строго в JSON:
{{
  "contacts": {{
    "name": "имя кандидата",
    "sex": "пол кандидата", 
    "city": "город",
    "number": "телефон",
    "email": "почта",
    "social": ["список социальных сетей"],
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
        "company": "название компании (обычно идет после указания периода работы в компании и общего времени)", 
        "sphere": "сектор, сфера, в которой работает компания", 
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
}}"""),
    ("human", "Проанализируй это резюме: {resume_text}")
])

UC_MATCHING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — HR-эксперт по подбору персонала. 
Твоя задача: сопоставить требования вакансии с данными кандидата и оценить соответствие.

Правила оценки:
- Оценивай только на основе данных из резюме
- Будь объективным и последовательным
- Учитывай как точные совпадения, так и близкие соответствия
- Выявляй критические несоответствия

Критерии оценки с весами:
1. Соответствие должности (job_title) - {job_title_weight}
2. Образование (education) - {education_weight}
3. Опыт работы (work_experience) - {experience_weight}
4. График работы (work_schedule) - {schedule_weight}
5. Формат работы (work_format) - {format_weight}
6. Дополнительные требования (additional_requirements) - {additional_weight}


ВЕРНИ ТОЛЬКО JSON БЕЗ ЛИШНИХ КОММЕНТАРИЕВ.

Формат JSON:
{{
  "candidate_info": {{
    "candidate_id": "{candidate_id}",
    "current_position": "текущая должность"
  }},
  "matching_results": {{
    "match_breakdown": {{
      "job_title_match": {{
        "score": 0-100,
        "weight": {job_title_weight},
        "weighted_score": 0-{job_title_weight},
        "explanation": "обоснование оценки"
      }},
      "education_match": {{
        "score": 0-100,
        "weight": {education_weight},
        "weighted_score": 0-{education_weight},
        "explanation": "обоснование оценки"
      }},
      "experience_match": {{
        "score": 0-100,
        "weight": {experience_weight},
        "weighted_score": 0-{experience_weight},
        "explanation": "обоснование оценки"
      }},
      "schedule_match": {{
        "score": 0-100,
        "weight": {schedule_weight},
        "weighted_score": 0-{schedule_weight},
        "explanation": "обоснование оценки"
      }},
      "format_match": {{
        "score": 0-100,
        "weight": {format_weight},
        "weighted_score": 0-{format_weight},
        "explanation": "обоснование оценки"
      }},
      "additional_match": {{
        "score": 0-100,
        "weight": {additional_weight},
        "weighted_score": 0-{additional_weight},
        "explanation": "обоснование оценки"
      }}
    "overall_score": 0-100,
    "is_suitable": true/false,
    "critical_issues": ["критические несоответствия"],
    }}
  }},
  "recommendation": {{
    "level": "рекомендован/условно рекомендован/не рекомендован",
    "reason": "обоснование рекомендации",
    "suggested_salary": "предлагаемая зарплата на основе опыта",
    "interview_priority": "высокий/средний/низкий"
  }}
}}"""),
    ("human", """ТРЕБОВАНИЯ ВАКАНСИИ:
Должность: {job_title}
Образование: {education}
Требуемый опыт: {work_experience} лет
Желаемая зарплата: {desired_salary} руб.
График работы: {work_schedule}
Формат работы: {work_format}
Дополнительные требования: {additional_requirements}

ДАННЫЕ КАНДИДАТА:
{resume_analysis}

Проведи сопоставление и верни оценку соответствия.""")
])

UC_QUESTION_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — эксперт по проведению интервью.
Твоя задача: сгенерировать персонализированные вопросы для собеседования.

Правила генерации:
- Сгенерируй 8-12 вопросов разного типа
- Вопросы должны быть основаны на анализе резюме
- Учитывай пробелы в навыках и опыт кандидата
- Адаптируй сложность вопросов под уровень кандидата

Формат вывода строго в JSON:
{{
  "interview_plan": {{
    "duration_minutes": "60",
    "structure": [
      {{
        "section": "введение",
        "time_allocation": "5 минут",
        "purpose": "знакомство и разогрев"
      }}
    ]
  }},
  "questions": [
    {{
      "type": "технический/поведенческий/кейсовый/мотивационный/культурный",
      "question": "текст вопроса",
      "purpose": "что проверяет этот вопрос",
      "expected_answer_indicators": ["признаки хорошего ответа"],
      "time_estimate": "1-2 минуты"
    }}
  ]
}}"""),
    ("human", """Проанализированное резюме кандидата: {resume_analysis}
Требования вакансии: {vacancy_requirements}
Дополнительные указания: {additional_instructions}

Сгенерируй персонализированные вопросы для интервью, учитывая:
- Пробелы в навыках
- Сильные стороны
- Опыт работы""")
])

import sys
import os

# Добавляем путь для импорта Langchain
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from ml.prompt_templates import ( 
    UC_MATCHING_PROMPT, 
    UC_QUESTION_GENERATION_PROMPT,
    UC_ANALYSIS_PROMPT
)

from langchain_core.prompts import ChatPromptTemplate

# Сохраняем как ChatPromptTemplate объекты
ANALYSIS_PROMPTS_CHAT = {
    "original": UC_ANALYSIS_PROMPT,
    "variant_a": ChatPromptTemplate.from_messages([
        ("system", """Ты — опытный HR-специалист по анализу резюме..."""),
        ("human", """Резюме кандидата:
{resume_text}

Извлеки структурированные данные в указанном JSON формате.""")
    ]),
    "variant_b": ChatPromptTemplate.from_messages([
        ("system", """Ты — AI-ассистент для парсинга резюме..."""),
        ("human", """Пожалуйста, проанализируй это резюме и верни структурированные данные:
{resume_text}

Верни только JSON без дополнительных комментариев.""")
    ])
}

# Создаем словарную версию для совместимости со старым кодом
ANALYSIS_PROMPTS = {
    "original": {
        "system": UC_ANALYSIS_PROMPT.messages[0].content if hasattr(UC_ANALYSIS_PROMPT.messages[0], 'content') else str(UC_ANALYSIS_PROMPT.messages[0]),
        "human": UC_ANALYSIS_PROMPT.messages[1].content if hasattr(UC_ANALYSIS_PROMPT.messages[1], 'content') else str(UC_ANALYSIS_PROMPT.messages[1])
    },
    "variant_a": {
        "system": """Ты — опытный HR-специалист по анализу резюме.
Твоя задача: извлечь структурированную информацию из текста резюме.

ПРАВИЛА АНАЛИЗА:
1. Извлекай только явно указанную информацию
2. Контакты — указывай только если явно указаны в резюме
3. Навыки — только те, что перечислены кандидатом
4. Опыт — рассчитывай общий стаж из указанных периодов работы
5. Образование — все указанные учебные заведения и степени
6. Не добавляй информацию, которой нет в тексте

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
        "sphere": "сфера деятельности компании", 
        "years": "число лет в компании",
        "description": "описание обязанностей и достижений"
      }}
    ]
  }},
  "education": [
    {{
      "institution": "учебное заведение",
      "degree": "степень или уровень образования",
      "year": "год окончания",
      "field": "специальность"
    }}
  ]
}}""",
        "human": """Резюме кандидата:
{resume_text}

Извлеки структурированные данные в указанном JSON формате."""
    },
    "variant_b": {
        "system": """Ты — AI-ассистент для парсинга резюме.
Твоя задача: преобразовать неструктурированный текст резюме в структурированный JSON.

ПРАВИЛА ИЗВЛЕЧЕНИЯ:
- Извлекай только факты, присутствующие в тексте
- Для навыков указывай только явно упомянутые
- Для опыта работы указывай подтвержденные периоды
- Для образования — только указанные учреждения
- Будь максимально точным и объективным

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
        "sphere": "сфера деятельности компании", 
        "years": "число лет",
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
        "human": """Пожалуйста, проанализируй это резюме и верни структурированные данные:
{resume_text}

Верни только JSON без дополнительных комментариев."""
    }
}

# Аналогично для MATCHING_PROMPTS и QUESTION_GENERATION_PROMPTS
MATCHING_PROMPTS = {
    "original": {
        "system": UC_MATCHING_PROMPT.messages[0].content if hasattr(UC_MATCHING_PROMPT.messages[0], 'content') else str(UC_MATCHING_PROMPT.messages[0]),
        "human": UC_MATCHING_PROMPT.messages[1].content if hasattr(UC_MATCHING_PROMPT.messages[1], 'content') else str(UC_MATCHING_PROMPT.messages[1])
    },
    "variant_a": {
        "system": """Ты — эксперт по оценке соответствия кандидатов.
Твоя задача: сравнить профиль кандидата с требованиями вакансии.

ПРАВИЛА ОЦЕНКИ:
- Оценивай только на основе предоставленных данных
- Используй взвешенную систему оценки
- Выявляй критические несоответствия
- Будь объективным и последовательным

КРИТЕРИИ ОЦЕНКИ С ВЕСАМИ:
1. Соответствие должности — {job_title_weight}%
2. Образование — {education_weight}%
3. Опыт работы — {experience_weight}%
4. График работы — {schedule_weight}%
5. Формат работы — {format_weight}%
6. Дополнительные требования — {additional_weight}%

ВЕРНИ ТОЛЬКО JSON БЕЗ ЛИШНИХ КОММЕНТАРИЕВ.

Формат JSON:
{{
  "matching_analysis": {{
    "candidate_summary": {{
      "current_position": "текущая должность",
      "experience_years": "опыт работы",
      "education_level": "уровень образования"
    }},
    "match_scores": {{
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
    }},
    "overall_score": 0-100,
    "is_suitable": true/false,
    "critical_issues": ["список критических несоответствий"]
  }},
  "recommendation": {{
    "decision": "рекомендован/условно рекомендован/не рекомендован",
    "reason": "обоснование решения",
    "suggested_salary": "предлагаемая зарплата",
    "interview_priority": "высокий/средний/низкий"
  }}
}}""",
        "human": """ТРЕБОВАНИЯ ВАКАНСИИ:
Должность: {job_title}
Образование: {education}
Требуемый опыт: {work_experience} лет
Желаемая зарплата: {desired_salary} руб.
График работы: {work_schedule}
Формат работы: {work_format}
Дополнительные требования: {additional_requirements}

ПРОФИЛЬ КАНДИДАТА:
{resume_analysis}

Проведи оценку соответствия кандидата требованиям вакансии."""
    },
    "variant_b": {
        "system": """Ты — система оценки соответствия кандидатов.
Твоя задача: провести комплексную оценку match между кандидатом и вакансией.

ПРАВИЛА КОМПЛЕКСНОЙ ОЦЕНКИ:
- Используй взвешенную систему оценки по критериям
- Выявляй критические несоответствия
- Учитывай как точные совпадения, так и близкие соответствия
- Дай обоснованную рекомендацию по найму

ВЕРНИ ТОЛЬКО JSON БЕЗ ЛИШНИХ КОММЕНТАРИЕВ.

Формат JSON:
{{
  "evaluation_results": {{
    "vacancy_requirements": {{
      "job_title": "{job_title}",
      "required_education": "{education}",
      "required_experience": "{work_experience} лет",
      "work_schedule": "{work_schedule}",
      "work_format": "{work_format}"
    }},
    "candidate_assessment": {{
      "match_score": 0-100,
      "strengths": ["сильные стороны кандидата"],
      "weaknesses": ["слабые стороны и пробелы"],
      "compatibility_analysis": "анализ совместимости"
    }},
    "detailed_breakdown": [
      {{
        "criterion": "должность",
        "match_level": "полный/частичный/отсутствует",
        "score": 0-100,
        "explanation": "обоснование"
      }},
      {{
        "criterion": "образование",
        "match_level": "полный/частичный/отсутствует",
        "score": 0-100,
        "explanation": "обоснование"
      }},
      {{
        "criterion": "опыт",
        "match_level": "полный/частичный/отсутствует",
        "score": 0-100,
        "explanation": "обоснование"
      }},
      {{
        "criterion": "график",
        "match_level": "полный/частичный/отсутствует",
        "score": 0-100,
        "explanation": "обоснование"
      }},
      {{
        "criterion": "формат",
        "match_level": "полный/частичный/отсутствует",
        "score": 0-100,
        "explanation": "обоснование"
      }}
    ]
  }},
  "hiring_recommendation": {{
    "recommendation": "нанять/рассмотреть/отклонить",
    "confidence_level": "высокий/средний/низкий",
    "reasoning": "подробное обоснование",
    "next_steps": "рекомендуемые следующие шаги"
  }}
}}""",
        "human": """ТРЕБОВАНИЯ ВАКАНСИИ:
- Должность: {job_title}
- Образование: {education}
- Требуемый опыт: {work_experience} лет
- Желаемая зарплата: {desired_salary} руб.
- График работы: {work_schedule}
- Формат работы: {work_format}
- Дополнительные требования: {additional_requirements}

АНАЛИЗ КАНДИДАТА:
{resume_analysis}

Рассчитай score соответствия и дай рекомендацию по найму."""
    }
}

QUESTION_GENERATION_PROMPTS = {
    "original": {
        "system": UC_QUESTION_GENERATION_PROMPT.messages[0].content if hasattr(UC_QUESTION_GENERATION_PROMPT.messages[0], 'content') else str(UC_QUESTION_GENERATION_PROMPT.messages[0]),
        "human": UC_QUESTION_GENERATION_PROMPT.messages[1].content if hasattr(UC_QUESTION_GENERATION_PROMPT.messages[1], 'content') else str(UC_QUESTION_GENERATION_PROMPT.messages[1])
    },
    "variant_a": {
        "system": """Ты — рекрутер, готовящий вопросы для собеседования.
Твоя задача: создать персонализированные вопросы на основе анализа кандидата.

ПРАВИЛА ГЕНЕРАЦИИ ВОПРОСОВ:
1. Создай 10-15 вопросов разного типа
2. Вопросы должны быть основаны на данных из резюме
3. Учитывай пробелы в навыках и несоответствия
4. Адаптируй сложность вопросов под уровень кандидата
5. Включи вопросы для проверки заявленного опыта

ТИПЫ ВОПРОСОВ:
- Технические (проверка конкретных навыков)
- Поведенческие (опыт, кейсы, ситуации)
- Мотивационные (цели, интересы, карьерные планы)
- Культурные (ценности, работа в команде, адаптация)
- Кейсовые (решение практических задач)

ВЕРНИ ТОЛЬКО JSON БЕЗ ЛИШНИХ КОММЕНТАРИЕВ.

Формат JSON:
{{
  "interview_plan": {{
    "duration_minutes": "60",
    "structure": [
      {{
        "section": "введение",
        "time_allocation": "5 минут",
        "purpose": "знакомство и создание комфортной атмосферы"
      }},
      {{
        "section": "техническая часть", 
        "time_allocation": "25 минут",
        "purpose": "оценка технических навыков и опыта"
      }},
      {{
        "section": "поведенческая часть",
        "time_allocation": "20 минут", 
        "purpose": "оценка soft skills и культурного соответствия"
      }},
      {{
        "section": "завершение",
        "time_allocation": "10 минут",
        "purpose": "ответы на вопросы кандидата и обратная связь"
      }}
    ]
  }},
  "questions": [
    {{
      "type": "технический/поведенческий/кейсовый/мотивационный/культурный",
      "question": "текст вопроса",
      "purpose": "что конкретно проверяет этот вопрос",
      "target_skill": "навык или качество, которое оценивается",
      "expected_answer_indicators": ["признаки хорошего ответа"],
      "follow_up_suggestions": ["возможные уточняющие вопросы"],
      "time_estimate": "1-3 минуты"
    }}
  ],
  "interviewer_guidelines": {{
    "focus_areas": ["ключевые области для оценки"],
    "red_flags": ["потенциальные тревожные сигналы"],
    "strengths_to_confirm": ["сильные стороны для подтверждения"]
  }}
}}""",
        "human": """КАНДИДАТ:
{resume_analysis}

ТРЕБОВАНИЯ ВАКАНСИИ:
{vacancy_requirements}

ДОПОЛНИТЕЛЬНЫЕ УКАЗАНИЯ:
{additional_instructions}

Сгенерируй персонализированные вопросы для собеседования."""
    },
    "variant_b": {
        "system": """Ты — помощник по подготовке к интервью.
Твоя задача: создать вопросы для комплексной оценки кандидата.

ФОКУСНЫЕ ОБЛАСТИ ОЦЕНКИ:
1. Проверка заявленных навыков и компетенций
2. Оценка реального опыта и достижений
3. Понимание мотивации и карьерных целей
4. Культурное соответствие и ценности
5. Потенциал роста и развития

ПРАВИЛА:
- Вопросы должны быть конкретными и измеримыми
- Учитывай уровень позиции (junior/middle/senior)
- Балансируй между техническими и поведенческими вопросами
- Включи вопросы для проверки пробелов в резюме

ВЕРНИ ТОЛЬКО JSON БЕЗ ЛИШНИХ КОММЕНТАРИЕВ.

Формат JSON:
{{
  "interview_metadata": {{
    "candidate_level": "junior/middle/senior",
    "interview_type": "техническое/комплексное/культурное",
    "total_questions": "число",
    "estimated_duration": "60-90 минут"
  }},
  "question_categories": [
    {{
      "category": "технические навыки",
      "weight": "40%",
      "questions": [
        {{
          "question": "текст вопроса",
          "skill_assessed": "конкретный навык",
          "difficulty": "легкий/средний/сложный",
          "purpose": "что проверяет",
          "evaluation_criteria": ["критерии оценки ответа"]
        }}
      ]
    }},
    {{
      "category": "опыт работы",
      "weight": "30%", 
      "questions": [
        {{
          "question": "текст вопроса",
          "experience_area": "область опыта",
          "purpose": "что проверяет",
          "expected_details": ["ожидаемые детали в ответе"]
        }}
      ]
    }},
    {{
      "category": "soft skills и культурное соответствие",
      "weight": "30%",
      "questions": [
        {{
          "question": "текст вопроса",
          "skill_assessed": "качество или компетенция",
          "purpose": "что проверяет",
          "behavioral_indicators": ["индикаторы поведения"]
        }}
      ]
    }}
  ],
  "scoring_guidelines": {{
    "scoring_scale": "1-5 (1 - плохо, 5 - отлично)",
    "evaluation_areas": ["список областей для оценки"],
    "decision_framework": "критерии для принятия решения"
  }}
}}""",
        "human": """АНАЛИЗ КАНДИДАТА:
{resume_analysis}

КОНТЕКСТ ВАКАНСИИ:
{vacancy_requirements}

УТОЧНЕНИЯ:
{additional_instructions}

Создай структурированный набор вопросов для оценки кандидата."""
    }
}
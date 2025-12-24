from langchain_core.prompts import ChatPromptTemplate

# =========================
# ANALYSIS PROMPTS
# =========================

UC_ANALYSIS_PROMPTS = {
    "baseline": ChatPromptTemplate.from_messages([
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
]),

    "variant_a": ChatPromptTemplate.from_messages([
        ("system", """Извлеки структурированные данные из резюме. Используй только предоставленный текст. Если данных нет — оставляй поле пустым или null.

Верни JSON следующей структуры без комментариев:
{{
  "contacts": {{"name": "", "sex": "", "city": "", "number": "", "email": "", "social": []}},
  "skills": {{"technical": [], "soft": [], "languages": []}},
  "experience": {{"total_years": 0, "relevant_years": 0, "positions": []}},
  "education": []
}}"""),
        ("human", "Резюме: {resume_text}")
    ]),

    "variant_b": ChatPromptTemplate.from_messages([
        ("system", """Анализируй резюме и заполняй JSON. Будь точен. Не придумывай."""),
        ("human", "Заполни JSON данными из резюме ниже:\n{resume_text}\n\nJSON структура:\n{{contacts: {{name: , sex: , city: , number: , email: , social: []}}, skills: {{technical: [], soft: [], languages: []}}, experience: {{total_years: 0, relevant_years: 0, positions: []}}, education: []}}")
    ])
}

# =========================
# MATCHING PROMPTS
# =========================

UC_MATCHING_PROMPTS = {
    "baseline": ChatPromptTemplate.from_messages([
        ("system", """Ты — HR-эксперт по подбору персонала. 
Твоя задача: сопоставить требования вакансии с данными кандидата и оценить соответствие.

Правила оценки:
- Оценивай только на основе данных из резюме
- Будь объективным и последовательным
- Учитывай как точные совпадения, так и близкие соответствия
- Выявляй критические несоответствия

Критерии оценки с весами:
1. Соответствие должности (job_title) - 0.2
2. Образование (education) - 0.15
3. Опыт работы (work_experience) - 0.25
4. График работы (work_schedule) - 0.05
5. Формат работы (work_format) - 0.05
6. Дополнительные требования (additional_requirements) - 0.3


ВЕРНИ ТОЛЬКО JSON БЕЗ ЛИШНИХ КОММЕНТАРИЕВ.

Формат JSON:
{{
  "candidate_info": {{
    "current_position": "текущая должность"
  }},
  "matching_results": {{
    "match_breakdown": {{
      "job_title_match": {{
        "score": 0-100,
        "weight": 0.2,
        "weighted_score": score * weight,
        "explanation": "обоснование оценки"
      }},
      "education_match": {{
        "score": 0-100,
        "weight": 0.15,
        "weighted_score": score * weight,
        "explanation": "обоснование оценки"
      }},
      "experience_match": {{
        "score": 0-100,
        "weight": 0.25,
        "weighted_score": score * weight,
        "explanation": "обоснование оценки"
      }},
      "schedule_match": {{
        "score": 0-100,
        "weight": 0.05,
        "weighted_score": score * weight,
        "explanation": "обоснование оценки"
      }},
      "format_match": {{
        "score": 0-100,
        "weight": 0.05,
        "weighted_score": score * weight,
        "explanation": "обоснование оценки"
      }},
      "additional_match": {{
        "score": 0-100,
        "weight": 0.3,
        "weighted_score": score * weight,
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
{vacancy_data}

ДАННЫЕ КАНДИДАТА:
{resume_analysis}

Проведи сопоставление и верни оценку соответствия.""")
    ]),

    "variant_a": ChatPromptTemplate.from_messages([
        ("system", """Оцени кандидата для вакансии. Заполни JSON. Баллы ставь объективно на основе данных. Задай критерии и веса критериям, также обоснуй поставленные баллы."""),
        ("human", """Вакансия:{vacancy_data}
Кандидат: {resume_analysis}
Оцени и верни JSON.""")
    ]),

    "variant_b": ChatPromptTemplate.from_messages([
        ("system", """Сравни данные кандидата с требованиями вакансии. Рассчитай взвешенный итоговый балл. Укажи критичные расхождения."""),
        ("human", """Требуется: {vacancy_data}.""")
    ])
}

# =========================
# QUESTIONS PROMPTS
# =========================

UC_QUESTION_PROMPTS = {
    "baseline": ChatPromptTemplate.from_messages([
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

Сгенерируй персонализированные вопросы для интервью, учитывая:
- Пробелы в навыках
- Сильные стороны
- Опыт работы""")
    ]),

    "variant_a": ChatPromptTemplate.from_messages([
        ("system", """Создай 8-12 вопросов для интервью на основе резюме и вакансии. Вопросы должны быть конкретными и проверять опыт/навыки. Укажи тип и цель каждого."""),
        ("human", """Резюме: {resume_analysis}\nВакансия: {vacancy_requirements}\nСгенерируй вопросы.""")
    ]),

    "variant_b": ChatPromptTemplate.from_messages([
        ("system", """Сгенерируй персонализированный план интервью (60 мин). 8-12 вопросов. Фокус на выявлении соответствия вакансии и пробелов в опыте. Без общих вопросов."""),
        ("human", """Кандидат: {resume_analysis}\nТребования вакансии: {vacancy_requirements}\nДай вопросы.""")
    ])
}
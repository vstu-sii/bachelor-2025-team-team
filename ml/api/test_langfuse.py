import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import Gemma3Text
from ml.api.langfuse_integration import run_full_pipeline

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    gemma = Gemma3Text()

    vacancy_data = {
        "job_title": "Data Scientist",
        "education": "Высшее техническое",
        "work_experience": 3,
        "desired_salary": 200000,
        "work_schedule": "Полный день",
        "work_format": "Гибрид",
        "additional_requirements": "Опыт работы с Python и ML"
    }

    result = run_full_pipeline(gemma, file_path="data/17.txt", vacancy_data=vacancy_data)
    print(result)
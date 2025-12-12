import pika
import json
import logging
import time
from pathlib import Path

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import HRBaseline  # ваш класс

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MAX_RETRIES = 3
RETRY_DELAY = 2  # секунды

class HRWorker:
    def __init__(self):
        self.hr_system = HRBaseline()
    
    def process_evaluation_task(self, task_id: str, task_data: dict) -> dict:
        """Обработка задачи оценки кандидата с retry логикой."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self.hr_system.evaluate_candidate_match(
                    resume_analysis=task_data["resume_analysis"],
                    vacancy_data=task_data["vacancy_data"],
                    criteria_weights=task_data.get("criteria_weights"),
                    candidate_id=task_data.get("candidate_id")
                )
                if result and "error" not in result:
                    return result
                logging.warning(f"[{task_id}] Попытка {attempt}: ошибка {result.get('error')}")
            except Exception as e:
                logging.error(f"[{task_id}] Попытка {attempt}: исключение {e}")
            time.sleep(RETRY_DELAY)
        return {"error": f"Не удалось обработать задание после {MAX_RETRIES} попыток"}
    
    def process_questions_task(self, task_id: str, task_data: dict) -> dict:
        """Обработка задачи генерации вопросов."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self.hr_system.generate_interview_questions(
                    resume_analysis=task_data["resume_analysis"],
                    vacancy_requirements=task_data["vacancy_requirements"],
                    skill_gaps=task_data.get("skill_gaps"),
                    strengths=task_data.get("strengths"),
                    experience_summary=task_data.get("experience_summary")
                )
                if result and "error" not in result:
                    return result
                logging.warning(f"[{task_id}] Попытка {attempt}: ошибка {result.get('error')}")
            except Exception as e:
                logging.error(f"[{task_id}] Попытка {attempt}: исключение {e}")
            time.sleep(RETRY_DELAY)
        return {"error": f"Не удалось сгенерировать вопросы после {MAX_RETRIES} попыток"}

def callback(ch, method, properties, body):
    worker = HRWorker()
    
    try:
        message = json.loads(body)
        task_id = message["task_id"]
        task_type = message["task_type"]  # "evaluation" или "questions"
        task_data = message["task_data"]
        
        logging.info(f"[{task_id}] Получено задание типа: {task_type}")

        # Обработка в зависимости от типа задачи
        if task_type == "evaluation":
            result = worker.process_evaluation_task(task_id, task_data)
        elif task_type == "questions":
            result = worker.process_questions_task(task_id, task_data)
        else:
            result = {"error": f"Неизвестный тип задачи: {task_type}"}

        # Сохраняем результат
        result_path = Path(f"./hr_results/{task_id}_{task_type}.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "task_type": task_type,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "result": result
            }, f, ensure_ascii=False, indent=2)

        logging.info(f"[{task_id}] Результат сохранён в {result_path}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag)  # подтверждаем чтобы не зацикливалось

def main():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
        channel = connection.channel()
        
        # Объявляем очередь для HR-задач
        channel.queue_declare(queue="hr_tasks_queue")
        channel.basic_qos(prefetch_count=1)  # обрабатываем по одной задаче за раз
        channel.basic_consume(queue="hr_tasks_queue", on_message_callback=callback)
        
        logging.info(" [*] HR Worker запущен. Ожидание сообщений...")
        channel.start_consuming()
        
    except KeyboardInterrupt:
        logging.info("Остановка HR worker по Ctrl+C")
    except Exception as e:
        logging.error(f"Ошибка соединения с RabbitMQ: {e}")

if __name__ == "__main__":
    main()
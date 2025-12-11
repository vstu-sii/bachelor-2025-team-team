# backend/routers/hr_worker.py
import asyncio
import json
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml.models.baseline import HRBaseline
import aio_pika

class HRTaskWorker:
    def __init__(self):
        self.hr = None
    
    async def initialize(self):
        """Инициализация модели один раз при запуске"""
        self.hr = HRBaseline()
        await self.hr.init_client()
        logging.info("✅ HR модель инициализирована")
    
    async def shutdown(self):
        """Корректное завершение"""
        if self.hr:
            await self.hr.close_client()
            logging.info("✅ HR модель закрыта")
    
    async def process_message(self, message: dict) -> dict:
        """Основной обработчик сообщений"""
        task_type = message.get("task_type")
        task_id = message.get("task_id", "unknown")
        
        logging.info(f"📥 Получена задача {task_id}: {task_type}")
        
        handlers = {
            "extract_resume": self._handle_extract,
            "evaluate_candidate": self._handle_evaluate,
            "generate_questions": self._handle_generate_questions,
            "check_extraction": self._handle_check_extraction,
            "check_questions": self._handle_check_questions,
        }
        
        if task_type not in handlers:
            return {"error": f"Unknown task type: {task_type}"}
        
        try:
            result = await handlers[task_type](message.get("data", {}))
            return {"success": True, "task_id": task_id, "result": result}
        except Exception as e:
            logging.error(f"❌ Ошибка в задаче {task_id}: {e}")
            return {"error": str(e), "task_id": task_id}
    
    async def _handle_extract(self, data: dict):
        """Извлечение данных из резюме"""
        file_path = data["file_path"]
        return await self.hr.extract_data_from_resume(file_path)
    
    async def _handle_evaluate(self, data: dict):
        """Оценка кандидата"""
        return await self.hr.evaluate_candidate_match(
            data["resume_data"],
            data["vacancy_data"]
        )
    
    async def _handle_generate_questions(self, data: dict):
        """Генерация вопросов"""
        return await self.hr.generate_interview_questions(
            data["resume_data"],
            data.get("vacancy_data")
        )

async def main():
    worker = HRTaskWorker()
    await worker.initialize()
    
    try:
        # Подключаемся к RabbitMQ
        connection = await aio_pika.connect_robust("amqp://localhost/")
        channel = await connection.channel()
        
        # Создаем очередь
        queue = await channel.declare_queue(
            "hr_processing_queue",
            durable=True,
            arguments={
                'x-max-priority': 10  # Приоритеты задач
            }
        )
        
        logging.info("🚀 HR Worker запущен и готов к работе")
        
        async def on_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    result = await worker.process_message(body)
                    
                    # Можно отправить результат обратно в другую очередь
                    if "task_id" in result:
                        result_queue = f"results_{result['task_id']}"
                        # ... отправка результата
                        
                except Exception as e:
                    logging.error(f"Ошибка обработки сообщения: {e}")
        
        await queue.consume(on_message)
        
        # Ждем вечно
        await asyncio.Future()
        
    finally:
        await worker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
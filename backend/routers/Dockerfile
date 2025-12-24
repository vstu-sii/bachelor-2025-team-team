# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /routers

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn python-dotenv
# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники приложения
COPY . .

# Экспонируем порт FastAPI
#EXPOSE 8001
EXPOSE 8000
# Запуск приложения через uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
CMD ["uvicorn", "backend/routers/ml/api/server:app", "--host", "0.0.0.0", "--port", "8000"]
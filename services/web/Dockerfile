FROM python:3.11
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN pip install --upgrade pip && pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev
COPY . /app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
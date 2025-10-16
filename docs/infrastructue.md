Структура проекта
project-root/
├── frontend/              # Next.js / React frontend
├── backend/               # FastAPI / Node.js backend
├── llm-service/           # LLM API и Jupyter
├── db/                    # Инициализация БД
├── monitoring/            # Grafana, Prometheus, Langfuse, Loki
│   ├── docker-compose.yml
│   └── dashboards/
├── docker-compose.dev.yml
└── docs/
    └── infrastructure.md
Архитектура инфраструктуры
![Flow Diagram](infr1.png)
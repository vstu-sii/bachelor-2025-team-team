# Используем официальный Node.js образ
FROM node:20-alpine

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и package-lock.json для установки зависимостей
COPY package*.json ./

# Устанавливаем зависимости
RUN npm ci

# Копируем исходники приложения
COPY . .

# Собираем Next.js проект
RUN npm run build

# Экспонируем порт приложения
EXPOSE 3000

# Запуск фронтенда в production режиме
CMD ["npm", "start"]

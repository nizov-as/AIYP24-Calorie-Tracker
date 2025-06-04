FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Отключаем Git-проверки Ultralytics
ENV GIT_PYTHON_REFRESH=quiet
ENV YOLO_DISABLE_GIT=1

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем модель и исходный код
COPY models/ ./models/
COPY . .

CMD ["python", "bot.py"]

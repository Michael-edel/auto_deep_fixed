FROM python:3.11-slim

WORKDIR /app

# Чтобы некоторые пакеты ставились без сюрпризов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# По умолчанию запускаем обработку invoices
CMD ["python", "main.py", "invoices"]

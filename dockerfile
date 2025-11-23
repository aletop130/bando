FROM python:3.11-slim

WORKDIR /app

COPY requirements_celery.txt .

RUN pip install --no-cache-dir -r requirements_celery.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

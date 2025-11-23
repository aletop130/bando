FROM python:3.12-slim

WORKDIR /app

# Installa tutte le dipendenze (sia per FastAPI che per Celery)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Installa anche le dipendenze Celery specifiche (se ce ne sono)
COPY requirements_celery.txt .
RUN pip install -r requirements_celery.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Il comando verrà sovrascritto nel docker-compose
CMD ["bash"]
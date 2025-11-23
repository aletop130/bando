FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG SERVICE=worker
ENV SERVICE=${SERVICE}

COPY requirements-api.txt requirements-worker.txt ./

RUN pip install --upgrade pip && \
    if [ "$SERVICE" = "api" ]; then \
        pip install -r requirements-api.txt; \
    else \
        pip install -r requirements-worker.txt; \
    fi

COPY . .
CMD ["bash"]

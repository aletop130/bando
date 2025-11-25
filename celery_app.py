# backend/celery_app.py  (o dove ce l'hai tu)

from dotenv import load_dotenv
load_dotenv()

from celery import Celery
import os
import requests
import time

REGOLO_API_URL = "https://api.regolo.ai/v1/chat/completions"
REGOLO_KEY = os.getenv("REGOLO_KEY")
REGOLO_MODEL = "Llama-3.3-70B-Instruct"

if not REGOLO_KEY:
    print("[WARN] REGOLO_KEY non impostata nell'ambiente del worker Celery")


celery_app = Celery(
    "regolo_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    task_time_limit=2100,
    task_soft_time_limit=2000,
)


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {REGOLO_KEY}",
}

@celery_app.task(bind=True, max_retries=2)
def regolo_call(self, prompt: str) -> str:
    """Task che chiama Regolo con lo stesso schema che funziona nel tuo esempio."""
    data = {
        "model": REGOLO_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
    }

    print(f"[WORKER] Inizio chiamata Regolo (task_id={self.request.id})")
    start = time.perf_counter()

    try:
        resp = requests.post(
            REGOLO_API_URL,
            headers=headers,
            json=data,
            timeout=2000,
        )
        elapsed = time.perf_counter() - start
        print(f"[WORKER] Regolo risposta HTTP {resp.status_code} in {elapsed:.2f}s")

        resp.raise_for_status()

        # stesso parsing del codice che funziona
        response_json = resp.json()
        content = (
            response_json
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:
            print("[WORKER][WARN] Nessun content in choices[0].message.content, uso raw text")
            content = resp.text

        print(f"[WORKER] Contenuto estratto ({len(content)} caratteri)")
        return content

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start
        print(f"[WORKER][TIMEOUT] dopo {elapsed:.2f}s (task_id={self.request.id})")
        raise self.retry(exc=Exception("Timeout"), countdown=60)

    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start
        print(f"[WORKER][ERRORE HTTP] {e} dopo {elapsed:.2f}s (task_id={self.request.id})")
        # logga anche il body per debugging
        if e.response is not None:
            print(f"[WORKER][ERRORE BODY] {e.response.text}")
        raise self.retry(exc=e, countdown=30)

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[WORKER][ERRORE GENERALE] {e} dopo {elapsed:.2f}s (task_id={self.request.id})")
        raise

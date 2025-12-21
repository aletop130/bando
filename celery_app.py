# backend/celery_app.py  (o dove ce l'hai tu)

from dotenv import load_dotenv
load_dotenv()

from celery import Celery
import os
import requests
import json

REGOLO_API_URL = "https://api.regolo.ai/v1/chat/completions"
REGOLO_KEY = os.getenv("REGOLO_KEY")
REGOLO_MODEL = "gpt-oss-120b"

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

# backend/celery_app.py - MODIFICA QUESTA FUNZIONE SOLO
@celery_app.task(bind=True, max_retries=2)
def regolo_call(self, prompt: str, response_format: str = "text") -> str:
    """Task che chiama Regolo con supporto per JSON mode"""
    data = {
        "model": REGOLO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Abbassato per consistenza
    }
    
    # Aggiungi JSON mode se richiesto
    if response_format == "json":
        data["response_format"] = {"type": "json_object"}
    
    print(f"[WORKER] Inizio chiamata Regolo (task_id={self.request.id}, format={response_format})")
    
    try:
        resp = requests.post(
            REGOLO_API_URL,
            headers=headers,
            json=data,
            timeout=2000,
        )
        
        resp.raise_for_status()
        response_json = resp.json()
        
        # Gestione diversa per JSON mode vs text
        if response_format == "json":
            # Restituisci il JSON completo
            return json.dumps(response_json)
        else:
            # Modalità testo normale
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content if content else resp.text
            
    except Exception as e:
        print(f"[WORKER][ERRORE] {e}")
        raise self.retry(exc=e, countdown=30)

# db_state.py
import sqlite3
import json
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect('job_status.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS job_status (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                progress TEXT,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

def update_job_status(job_id: str, status: str, progress: str, data: dict = None):
    with get_db() as conn:
        conn.execute('''
            INSERT OR REPLACE INTO job_status (job_id, status, progress, data)
            VALUES (?, ?, ?, ?)
        ''', (job_id, status, progress, json.dumps(data or {})))
        conn.commit()

def get_job_status(job_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM job_status WHERE job_id = ?', 
            (job_id,)
        ).fetchone()
        
        if row:
            return {
                'job_id': row['job_id'],
                'status': row['status'],
                'progress': row['progress'],
                **json.loads(row['data'] or '{}')
            }
        return {}
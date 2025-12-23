import os
import uuid
import json
from typing import Optional, List, Dict
from pydantic import BaseModel
from celery import group
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery_app import process_document_pipeline
from typing import Union
from graph_rag_pipeline import (
    retrieve_graph_context,       # GraphRAG
    graphRAG_run_with_history,       
    
    neo4j_driver,
    qdrant_client,                # Clients
    VECTOR_DIM,
)

# Classi

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "Bandi"


class UploadResponse(BaseModel):
    job_id: str
    message: str
    status: str


class QueryResponse(BaseModel):
    answer: str
    graph_context: Optional[dict] = None
    entity_ids_found: Optional[List[str]] = None
    qdrant_texts_count: Optional[int] = None


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    error: Optional[str] = None
    details: Optional[dict] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    collection_name: Optional[str] = "Bandi"


class ChatResponse(BaseModel):
    message: ChatMessage
    graph_context: Optional[dict] = None
    entity_ids_found: Optional[List[str]] = None


# Inizializzazione Fast API e React Frontend

app = FastAPI(title="GraphRAG Bandi API - Ontology Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = "uploads"
os.makedirs(STORAGE_DIR, exist_ok=True)

processing_status = {}

@app.post("/upload")
async def upload_document(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    collection_name: str = "Bandi"
):
    """
    Endpoint per caricare uno o più documenti PDF.
    Supporta sia singolo file che lista di file.
    """
    # Normalizza input in lista
    if isinstance(file, UploadFile):
        file_list = [file]
        is_single = True
    else:
        file_list = file
        is_single = False
    
    if not file_list:
        raise HTTPException(status_code=400, detail="Nessun file caricato")
    
    # Verifica tutti i file
    for f in file_list:
        if not f.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {f.filename} non è un PDF")
    
    responses = []
    tasks = []
    batch_job_ids = []
    
    # Prepara tutti i file e tasks
    for f in file_list:
        job_id = str(uuid.uuid4())
        batch_job_ids.append(job_id)
        file_path = os.path.join(STORAGE_DIR, f"{job_id}_{f.filename}")
        
        # Salva il file
        with open(file_path, "wb") as file_obj:
            content = await f.read()
            file_obj.write(content)
        
        # Inizializza stato
        processing_status[job_id] = {
            "status": "queued", 
            "progress": "In coda su Celery...",
            "collection_name": collection_name,
            "filename": f.filename
        }
        
        # Crea task Celery
        task = process_document_pipeline.s(job_id, file_path, collection_name)
        tasks.append(task)
        
        # Prepara risposta per questo file
        response_data = {
            "job_id": job_id,
            "message": f"Documento caricato: {f.filename}",
            "status": "queued"
        }
        responses.append(response_data)
    
    # Avvia tutti i tasks in parallelo usando group
    if tasks:
        if len(tasks) == 1 and is_single:
            # Singolo file: usa delay() normale
            task = tasks[0]
            result = task.delay()
            processing_status[batch_job_ids[0]].update({
                "celery_task_id": result.id,
                "async_result_id": result.id
            })
        else:
            # Multipli file: usa group()
            job_group = group(tasks)
            group_result = job_group.apply_async()
            
            # Aggiorna ogni job con l'ID del task Celery
            for i, job_id in enumerate(batch_job_ids):
                if i < len(group_result.children):
                    processing_status[job_id].update({
                        "celery_task_id": group_result.children[i].id,
                        "async_result_id": group_result.children[i].id
                    })
    
    # Restituisci risposta appropriata
    if is_single:
        return responses[0]  # Singolo oggetto
    else:
        return responses  # Lista di oggetti


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Endpoint per controllare lo stato del processing"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job ID non trovato")
    
    status = processing_status[job_id]
    return StatusResponse(
        job_id=job_id,
        status=status.get("status", "unknown"),
        progress=status.get("progress"),
        error=status.get("error"),
        details=status.get("details")
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Chat con message history"""
    collection_name = request.collection_name or "Bandi"
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="La lista dei messaggi non può essere vuota")
    
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="L'ultimo messaggio deve essere dell'utente")
    
    user_query = last_message.content
    
    conversation_history = None
    if len(request.messages) > 1:
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages[:-1]
        ]
    
    try:
        # Verifica se la collection esiste
        try:
            qdrant_client.get_collection(collection_name)
        except Exception:
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="⚠️ Nessun documento processato trovato. Carica un documento PDF per iniziare."
                ),
                graph_context=None,
                entity_ids_found=None
            )
        
        # Recupero contesto
        entity_ids, qdrant_texts, graph_context = retrieve_graph_context(
            query=user_query,
            collection_name=collection_name,
            top_k=10
        )
        
        if not entity_ids:
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="Nessuna informazione rilevante trovata per la tua query. Prova a riformulare la domanda o a essere più specifico."
                ),
                graph_context=None,
                entity_ids_found=[]
            )
        
        # GraphRAG con history
        answer = graphRAG_run_with_history(graph_context, user_query, conversation_history)
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=answer
            ),
            graph_context=graph_context,
            entity_ids_found=entity_ids
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore durante la chat: {str(e)}")


@app.get("/collections")
async def list_collections():
    """Endpoint per verificare quali collection sono disponibili"""
    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [col.name for col in collections.collections],
            "default": "Bandi",
            "note": "Pipeline ontologica con collegamenti diretti tra chunk e nodi del grafo"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ontology/{job_id}")
async def get_ontology(job_id: str):
    """Endpoint per recuperare l'ontologia estratta"""
    ontology_file = os.path.join(STORAGE_DIR, f"{job_id}_ontology.json")
    
    if not os.path.exists(ontology_file):
        raise HTTPException(status_code=404, detail="Ontologia non trovata")
    
    try:
        with open(ontology_file, 'r') as f:
            ontology_data = json.load(f)
        
        return {
            "job_id": job_id,
            "ontology": ontology_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_jobs():
    """Endpoint per listare tutti i jobs con il loro stato"""
    return {
        "total_jobs": len(processing_status),
        "jobs": [
            {
                "job_id": job_id,
                "status": info.get("status"),
                "filename": info.get("filename"),
                "progress": info.get("progress")
            }
            for job_id, info in processing_status.items()
        ]
    }


@app.get("/health")
async def health_check():
    """Health check per verificare lo stato dei servizi"""
    try:
        # Verifica Neo4j
        neo4j_driver.verify_connectivity()
        neo4j_status = "healthy"
    except Exception as e:
        neo4j_status = f"error: {str(e)}"
    
    try:
        # Verifica Qdrant
        qdrant_client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return {
        "neo4j": neo4j_status,
        "qdrant": qdrant_status,
        "total_jobs": len(processing_status),
        "storage_dir": STORAGE_DIR
    }


@app.delete("/cleanup")
async def cleanup_old_jobs(days: int = 1):
    """Endpoint per pulire jobs vecchi di N giorni"""
    import time
    from datetime import datetime, timedelta
    
    cutoff_time = datetime.now() - timedelta(days=days)
    deleted_count = 0
    
    for job_id in list(processing_status.keys()):
        job_info = processing_status[job_id]
        
        # Verifica se il job è completato o fallito
        status = job_info.get("status")
        if status in ["completed", "error"]:
            # Controlla se è vecchio (semplice implementazione)
            deleted_count += 1
            del processing_status[job_id]
    
    return {
        "message": f"Puliti {deleted_count} jobs vecchi",
        "remaining_jobs": len(processing_status)
    }


@app.get("/")
async def root():
    return {
        "message": "GraphRAG Bandi API - Pipeline Ontologica",
        "version": "4.0",
        "feature": "Upload singolo con processing Celery",
        "endpoints": {
            "POST /upload": "Carica un singolo PDF e processa con pipeline ontologica",
            "GET /status/{job_id}": "Stato processing",
            "POST /chat": "Chat con memoria",
            "GET /ontology/{job_id}": "Visualizza ontologia estratta",
            "GET /collections": "Lista collections",
            "GET /jobs": "Lista tutti i jobs",
            "GET /health": "Stato servizi",
            "DELETE /cleanup": "Pulizia jobs vecchi"
        },
        "note": "Per processare più PDF, fare chiamate multiple a /upload"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
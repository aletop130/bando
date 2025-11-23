from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import time
from typing import Optional, List
import asyncio

from graph_rag_pipeline import (
    extract_pdf_text_with_tables,
    chunk_text,
    build_graph_from_chunks,
    ingest_to_neo4j,
    ingest_to_qdrant,
    create_collection,
    retriever_search,
    fetch_related_graph,
    format_graph_context,
    graphRAG_run,
    graphRAG_run_with_history,
    neo4j_driver,
    qdrant_client,
    VECTOR_DIM,
)

app = FastAPI(title="GraphRAG Bandi API")

# CORS per permettere chiamate dal frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = "uploads"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Stato globale per tracking processing
processing_status = {}


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


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    error: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" o "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    collection_name: Optional[str] = "Bandi"

class ChatResponse(BaseModel):
    message: ChatMessage
    graph_context: Optional[dict] = None


def process_document(job_id: str, pdf_path: str, collection_name: str = "Bandi"):
    """Processa il documento in background"""
    try:
        processing_status[job_id] = {"status": "processing", "progress": "Lettura PDF..."}
        
        # 1. Estrazione testo
        processing_status[job_id]["progress"] = "Estrazione testo dal PDF..."
        raw_data = extract_pdf_text_with_tables(pdf_path)
        
        # 2. Chunking
        processing_status[job_id]["progress"] = "Creazione chunk..."
        chunks = chunk_text(raw_data, max_words=800)
        
        if not chunks:
            processing_status[job_id] = {
                "status": "error",
                "error": "Nessun chunk estratto dal documento"
            }
            return
        
        # 3. Creazione collection Qdrant
        processing_status[job_id]["progress"] = "Creazione collection Qdrant..."
        create_collection(qdrant_client, collection_name, VECTOR_DIM)
        
        # 4. Estrazione grafo
        processing_status[job_id]["progress"] = "Estrazione grafo di conoscenza (questo può richiedere tempo)..."
        nodes, relationships, chunk_node_mapping = build_graph_from_chunks(chunks)
        
        # 5. Ingest Neo4j
        processing_status[job_id]["progress"] = "Salvataggio in Neo4j..."
        ingest_to_neo4j(nodes, relationships)
        
        # 6. Ingest Qdrant
        processing_status[job_id]["progress"] = "Salvataggio in Qdrant..."
        ingest_to_qdrant(collection_name, chunks, chunk_node_mapping)
        
        processing_status[job_id] = {
            "status": "completed",
            "progress": "Completato!",
            "collection_name": collection_name,
            "chunks_count": len(chunks),
            "nodes_count": len(nodes),
            "relationships_count": len(relationships)
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        processing_status[job_id] = {
            "status": "error",
            "error": error_msg
        }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Endpoint per caricare un documento PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo file PDF sono supportati")
    
    # Salva il file
    job_id = str(uuid.uuid4())
    file_path = os.path.join(STORAGE_DIR, f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Inizializza stato
    processing_status[job_id] = {"status": "queued", "progress": "In coda..."}
    
    # Avvia processing in background
    background_tasks.add_task(process_document, job_id, file_path)
    
    return UploadResponse(
        job_id=job_id,
        message=f"Documento caricato: {file.filename}",
        status="queued"
    )


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
        error=status.get("error")
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Endpoint per fare query sul documento processato"""
    collection_name = request.collection_name or "Bandi"
    
    try:
        # 1. Retrieval
        retriever_result = retriever_search(
            neo4j_driver,
            qdrant_client,
            collection_name,
            request.query
        )
        
        # 2. Estrai entity IDs
        entity_ids = []
        for item in getattr(retriever_result, "items", []):
            try:
                # Prova a estrarre l'ID dal content
                content_str = str(item.content)
                if "'id': '" in content_str:
                    entity_id = content_str.split("'id': '")[1].split("'")[0]
                    entity_ids.append(entity_id)
            except Exception:
                continue
        
        if not entity_ids:
            return QueryResponse(
                answer="Nessuna entità rilevante trovata per la tua query. Prova a riformulare la domanda.",
                graph_context=None
            )
        
        # 3. Fetch subgraph
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        
        # 4. Format context
        graph_context = format_graph_context(subgraph)
        
        # 5. GraphRAG
        answer = graphRAG_run(graph_context, request.query)
        
        return QueryResponse(
            answer=answer,
            graph_context=graph_context
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore durante la query: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """
    Endpoint per chat con il documento processato.
    Accetta una lista di messaggi (conversazione) e restituisce la risposta.
    """
    collection_name = request.collection_name or "Bandi"
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="La lista dei messaggi non può essere vuota")
    
    # L'ultimo messaggio è la query corrente
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="L'ultimo messaggio deve essere dell'utente")
    
    user_query = last_message.content
    
    # Prepara la storia della conversazione (escludendo l'ultimo messaggio)
    conversation_history = None
    if len(request.messages) > 1:
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages[:-1]
        ]
    
    try:
        # 1. Retrieval
        retriever_result = retriever_search(
            neo4j_driver,
            qdrant_client,
            collection_name,
            user_query
        )
        
        # 2. Estrai entity IDs
        entity_ids = []
        for item in getattr(retriever_result, "items", []):
            try:
                content_str = str(item.content)
                if "'id': '" in content_str:
                    entity_id = content_str.split("'id': '")[1].split("'")[0]
                    entity_ids.append(entity_id)
            except Exception:
                continue
        
        if not entity_ids:
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="Nessuna entità rilevante trovata per la tua query. Prova a riformulare la domanda o a essere più specifico."
                ),
                graph_context=None
            )
        
        # 3. Fetch subgraph
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        
        # 4. Format context
        graph_context = format_graph_context(subgraph)
        
        # 5. GraphRAG con storia conversazionale
        answer = graphRAG_run_with_history(graph_context, user_query, conversation_history)
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=answer
            ),
            graph_context=graph_context
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore durante la chat: {str(e)}")


@app.get("/collections")
async def list_collections():
    """Endpoint per verificare quali collection sono disponibili"""
    try:
        # Qdrant non ha un metodo diretto per listare, ma possiamo provare a verificare
        # In alternativa, potresti mantenere una lista in memoria o in DB
        return {
            "message": "Usa il nome della collection usata durante l'upload",
            "default": "Bandi"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "GraphRAG Bandi API", "version": "1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

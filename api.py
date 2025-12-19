from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import time
from typing import Optional, List
import asyncio

# Import delle nuove funzioni con attributi dei nodi
from graph_rag_pipeline import (
    extract_pdf_text_with_tables,
    chunk_text,
    build_graph_from_chunks,
    ingest_to_neo4j,
    ingest_to_qdrant,
    create_collection,
    # FUNZIONI AGGIORNATE
    retriever_search,        # Ora restituisce 4 valori: (retriever_result, qdrant_texts, entity_ids, qdrant_node_mappings)
    fetch_related_graph,     # Ora include attributi dei NODI
    format_graph_context,    # Ora formatta attributi dei NODI
    graphRAG_run,            # Prompt aggiornato per attributi nodi
    graphRAG_run_with_history,  # Prompt aggiornato per attributi nodi
    normalize_whitespace,
    neo4j_driver,
    qdrant_client,
    VECTOR_DIM,
)

app = FastAPI(title="GraphRAG Bandi API")

# CORS per permettere chiamate dal frontend React
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
    # Aggiungiamo info su cosa è stato trovato
    entity_ids_found: Optional[List[str]] = None
    qdrant_texts_count: Optional[int] = None


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    error: Optional[str] = None


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


def process_document(job_id: str, pdf_path: str, collection_name: str = "Bandi"):
    """Processa il documento in background"""
    try:
        processing_status[job_id] = {"status": "processing", "progress": "Lettura PDF..."}
        
        processing_status[job_id]["progress"] = "Estrazione testo dal PDF..."
        raw_data = extract_pdf_text_with_tables(pdf_path)
        clean_data = normalize_whitespace(raw_data)
        
        processing_status[job_id]["progress"] = "Creazione chunk..."
        chunks = chunk_text(clean_data, max_words=250, overlap_words=35)
        
        if not chunks:
            processing_status[job_id] = {
                "status": "error",
                "error": "Nessun chunk estratto dal documento"
            }
            return
        
        processing_status[job_id]["progress"] = "Creazione collection Qdrant..."
        create_collection(qdrant_client, collection_name, VECTOR_DIM)
        
        processing_status[job_id]["progress"] = "Estrazione grafo di conoscenza..."
        nodes, relationships, chunk_node_mapping = build_graph_from_chunks(chunks)
        
        processing_status[job_id]["progress"] = "Salvataggio in Neo4j..."
        ingest_to_neo4j(nodes, relationships)
        
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
    
    job_id = str(uuid.uuid4())
    file_path = os.path.join(STORAGE_DIR, f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    processing_status[job_id] = {"status": "queued", "progress": "In coda..."}
    
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
        # 1. Retrieval - USIAMO LA NUOVA VERSIONE che restituisce 4 valori
        retriever_result, qdrant_texts, entity_ids, qdrant_node_mappings = retriever_search(
            neo4j_driver,
            qdrant_client,
            collection_name,
            request.query
        )
        
        if not entity_ids:
            return QueryResponse(
                answer="Nessuna entità rilevante trovata per la tua query. Prova a riformulare la domanda.",
                graph_context=None,
                entity_ids_found=[],
                qdrant_texts_count=0
            )
        
        # 2. Fetch subgraph - USIAMO LA NUOVA VERSIONE con attributi dei nodi
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        
        # 3. Format context - USIAMO LA NUOVA VERSIONE con attributi dei nodi
        graph_context = format_graph_context(subgraph, qdrant_texts)
        
        # 4. GraphRAG - USIAMO LA NUOVA VERSIONE che sa gestire attributi dei nodi
        answer = graphRAG_run(graph_context, request.query)
        
        return QueryResponse(
            answer=answer,
            graph_context=graph_context,
            entity_ids_found=entity_ids,
            qdrant_texts_count=len(qdrant_texts)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore durante la query: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """
    Endpoint per chat con il documento processato.
    Ora con supporto per attributi dei nodi nel grafo.
    """
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
        
        # 1. Retrieval - NUOVA VERSIONE con 4 valori restituiti
        retriever_result, qdrant_texts, entity_ids, qdrant_node_mappings = retriever_search(
            neo4j_driver,
            qdrant_client,
            collection_name,
            user_query
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
        
        # 2. Fetch subgraph - NUOVA VERSIONE con attributi dei nodi
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        
        # 3. Format context - NUOVA VERSIONE con attributi dei nodi
        graph_context = format_graph_context(subgraph, qdrant_texts)
        
        # 4. GraphRAG con storia conversazionale - NUOVA VERSIONE che gestisce attributi nodi
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
        # Potremmo voler tenere traccia delle collection create
        # Per ora restituiamo un messaggio generico
        return {
            "message": "Usa il nome della collection usata durante l'upload",
            "default": "Bandi",
            "note": "Il sistema supporta attributi dei nodi nel grafo di conoscenza"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Endpoint per verificare lo stato dei servizi"""
    try:
        neo4j_healthy = False
        qdrant_healthy = False
        
        # Verifica Neo4j
        try:
            with neo4j_driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    neo4j_healthy = True
        except:
            pass
        
        # Verifica Qdrant
        try:
            collections = qdrant_client.get_collections()
            qdrant_healthy = True
        except:
            pass
        
        return {
            "neo4j": "healthy" if neo4j_healthy else "unhealthy",
            "qdrant": "healthy" if qdrant_healthy else "unhealthy",
            "api": "healthy",
            "features": "graph_with_node_attributes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "GraphRAG Bandi API con attributi dei nodi",
        "version": "2.0",
        "feature": "Graph con attributi dei nodi invece che delle relazioni"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
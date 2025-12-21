import os
import uuid
import json
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from graph_rag_pipeline import (
    
    extract_pdf_text_with_tables,
    normalize_whitespace,         # Estrazione testo da pdf
    chunk_text,
    
    extract_ontology_from_text,   
    create_ontology_graph,        # Ontologia e Neo4j Ingestion  
    ingest_to_neo4j,                 
    
    create_collection,
    ingest_to_qdrant,             # Chunking e Qdrant Ingestion
    
    retrieve_graph_context,       #GraphRAG
    graphRAG_run_with_history,       
    
    neo4j_driver,
    qdrant_client,                #Clients
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


def process_document_ontology(job_id: str, pdf_path: str, collection_name: str = "Bandi"):
    """NUOVA PIPELINE: Processa il documento usando ontologia (1 chiamata LLM)"""
    try:
        processing_status[job_id] = {"status": "processing", "progress": "Lettura PDF..."}
        
        # 1. Estrazione testo
        processing_status[job_id]["progress"] = "Estrazione testo dal PDF..."
        raw_data = extract_pdf_text_with_tables(pdf_path)
        clean_data = normalize_whitespace(raw_data)
        
        # 2. Estrazione ontologia (1 chiamata LLM)
        processing_status[job_id]["progress"] = "Estrazione ontologia strutturata (1 chiamata LLM)..."
        ontology = extract_ontology_from_text(clean_data)
        ontology.file_name = os.path.basename(pdf_path)
        
        processing_status[job_id]["details"] = {
            "identificativo": ontology.identificativo,
            "dimensioni_ammesse": ontology.dimensioni_ammesse,
            "tipologie_intervento": ontology.tipologie_intervento,
            "criteri_count": len(ontology.criteri_selezione)
        }
        
        # 3. Creazione grafo dall'ontologia
        processing_status[job_id]["progress"] = "Creazione grafo ontologico..."
        nodes, relationships, bando_attrs = create_ontology_graph(ontology) 
        
        # 4. Ingestion in Neo4j
        processing_status[job_id]["progress"] = "Salvataggio in Neo4j..."

        ingest_to_neo4j(nodes, relationships, bando_attrs=bando_attrs)

        # 5. Chunking del testo per Qdrant
        processing_status[job_id]["progress"] = "Creazione chunk per ricerca semantica..."
        chunks = chunk_text(clean_data, max_words=250, overlap_words=35)
        
        # 6. Crea/verifica collection Qdrant
        processing_status[job_id]["progress"] = "Configurazione Qdrant..."
        create_collection(qdrant_client, collection_name, VECTOR_DIM)
        
        # 7. Ingestione in Qdrant con collegamenti ai nodi Neo4j
        processing_status[job_id]["progress"] = "Salvataggio in Qdrant con collegamenti al grafo..."
        points_count = ingest_to_qdrant(
            collection_name=collection_name,
            chunks=chunks,
            ontology=ontology,
            nodes_dict=nodes
        )
        
        # 8. Salva metadata
        processing_status[job_id] = {
            "status": "completed",
            "progress": "Completato!",
            "collection_name": collection_name,
            "chunks_count": len(chunks),
            "nodes_count": len(nodes),
            "relationships_count": len(relationships),
            "qdrant_points": points_count,
            "ontology": {
                "identificativo": ontology.identificativo,
                "autorita": ontology.autorita,
                "dotazione": ontology.dotazione_finanziaria,
                "interventi": len(ontology.interventi),
                "criteri": len(ontology.criteri_selezione)
            },
            "bando_attributes": bando_attrs  
        }
        
        # Salva l'ontologia per riferimento
        ontology_file = os.path.join(STORAGE_DIR, f"{job_id}_ontology.json")
        with open(ontology_file, 'w') as f:
            json.dump(ontology.model_dump_json(), f, indent=2, default=str)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"[ERROR] Process failed: {error_msg}\n{traceback_str}")
        
        processing_status[job_id] = {
            "status": "error",
            "error": error_msg,
            "traceback": traceback_str
        }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = "Bandi"
):
    #Endpoint per caricare un documento PDF con NUOVA pipeline

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo file PDF sono supportati")
    
    job_id = str(uuid.uuid4())
    file_path = os.path.join(STORAGE_DIR, f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    processing_status[job_id] = {
        "status": "queued", 
        "progress": "In coda...",
        "collection_name": collection_name
    }
    
    #Ontology Pipelineù

    background_tasks.add_task(
        process_document_ontology, 
        job_id, 
        file_path, 
        collection_name
    )
    
    return UploadResponse(
        job_id=job_id,
        message=f"Documento caricato: {file.filename} (pipeline ontologica)",
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
        error=status.get("error"),
        details=status.get("details")
    )


@app.post("/chat", response_model=ChatResponse)

#Chat con message history

async def chat_with_document(request: ChatRequest):
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
        
        # 1. Recupera contesto usando la NUOVA funzione
        entity_ids, qdrant_texts, graph_context = retrieve_graph_context(
            query=user_query,
            collection_name=collection_name,
            top_k=5
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
        
        # 2. GraphRAG con history
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
            "features": "ontology_pipeline",
            "pipeline": "single_llm_call_ontology_extraction"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "GraphRAG Bandi API - Pipeline Ontologica",
        "version": "3.0",
        "feature": "Ontology extraction + Graph with node attributes + Qdrant-Neo4j linkage",
        "endpoints": {
            "POST /upload": "Carica PDF e processa con pipeline ontologica",
            "GET /status/{job_id}": "Stato processing",
            "POST /query": "Query singola sul bando",
            "POST /chat": "Chat con memoria",
            "GET /ontology/{job_id}": "Visualizza ontologia estratta",
            "GET /collections": "Lista collections",
            "GET /health": "Stato servizi"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
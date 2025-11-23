import os
import sys
import time
import uuid
import json
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient, models
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever

from celery_app import regolo_call

from typing import List, Optional


class GraphEntry(BaseModel):
    node: str
    target_node: str | None
    relationship: str | None


class GraphComponents(BaseModel):
    graph: list[GraphEntry]



# =========================
# ENV & CLIENTS
# =========================

load_dotenv()  # oppure load_dotenv('.env.local')

qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")


neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

if qdrant_key:
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key,
    )
else:
    qdrant_client = QdrantClient(
        url=qdrant_url,
    )



# Modello di embedding TESTO → VETTORE
# (puoi cambiarlo con un altro sentence transformer se vuoi)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM = embedding_model.get_sentence_embedding_dimension()


# =========================
# HELPER REGOL0
# =========================




# =========================
# PDF → TESTO (+ pseudo-tabelle)
# =========================

def build_internal_prompt(prompt: str) -> str:
    return f"""
Sei un estrattore estremamente preciso di relazioni per grafi di conoscenza.
DAL TESTO DEVI ESTRARRE IL MAGGIOR NUMERO POSSIBILE DI RELAZIONI CORRETTE TRA ENTITÀ.

Devi restituire ESCLUSIVAMENTE e SOLAMENTE SENZA ALTRE PAROLE ESTERNE un JSON con la seguente struttura ESATTA:

{{
  "graph": [
    {{
      "node": "Persona/Entità di partenza",
      "target_node": "Entità collegata",
      "relationship": "Tipo di relazione"
    }}
  ]
}}

REGOLE GENERALI:
- Usa ESATTAMENTE le chiavi: "graph", "node", "target_node", "relationship".
- "node" e "target_node" devono essere nomi brevi e leggibili di concetti/entità
  (es. "Articolo 2", "Beneficiari", "Requisiti di accesso", "Spesa ammissibile").
- "relationship" deve essere una breve etichetta testuale (in italiano o inglese)
  che descrive il tipo di relazione (es. "definisce", "regola", "applica a",
  "riguarda", "dipende da", "richiede", "fa parte di", "è un tipo di").

...

Testo da analizzare:
{prompt}
"""

def parse_graph(raw: str):
    """Estrae e valida il JSON dalla risposta di Regolo."""
    if not raw or not raw.strip():
        print("[PARSE] Risposta vuota")
        return []
    
    # estrazione robusta del JSON (come in sdsù)
    txt = raw.strip()
    if not txt.startswith("{"):
        i = txt.find("{")
        j = txt.rfind("}")
        if i != -1 and j != -1:
            txt = txt[i:j+1]
        else:
            print("[PARSE] Nessun JSON trovato nel testo")
            print(f"[PARSE] Raw ricevuto")
            return []
    
    try:
        parsed = GraphComponents.model_validate_json(txt)
        return parsed.graph
    except Exception as e:
        print(f"[PARSE ERROR] {type(e).__name__}: {e}")
        print(f"[PARSE] Raw ricevuto")
        print(f"[PARSE] Tentativo di estrazione")
        return []


def extract_pdf_text_with_tables(pdf_path: str) -> str:
    """
    Estrae tutto il testo del PDF, comprese le tabelle (in forma testuale),
    restituendo un'unica stringa pronta per il chunking.
    """
    doc = fitz.open(pdf_path)
    all_text_blocks = []

    for page_index in range(doc.page_count):
        page = doc[page_index]

        # Testo normale
        page_text = page.get_text("text")

        # Estrazione blocchi: (x0, y0, x1, y1, text, block_no, block_type, ...)
        blocks = page.get_text("blocks")
        table_texts = []

        for b in blocks:
            text_block = b[4]

            # Heuristica: se contiene colonne, pipe, tab → possibile tabella
            if "|" in text_block or "\t" in text_block:
                # Normalizzazione per aiutarci nel chunking
                cleaned = text_block.replace("\t", " | ")
                table_texts.append(cleaned)

        if table_texts:
            tables_serialized = "\n[TABELLA]:\n" + "\n".join(table_texts)
            full_page_text = page_text + "\n" + tables_serialized
        else:
            full_page_text = page_text

        all_text_blocks.append(full_page_text)

    doc.close()

    # Restituiamo *tutto* come unico grande testo
    return "\n\n".join(all_text_blocks)

def chunk_text(text: str, max_words: int = 400) -> list[str]:
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


# =========================
# ESTR. GRAFO CON REGOLO
# =========================

def regolo_llm_parser(text: str):
    internal_prompt = build_internal_prompt(text)
    task = regolo_call.delay(internal_prompt)
    return task




def extract_graph_components(task):
    """Estrae i componenti del grafo da un task Celery completato."""
    try:
        raw = task.get(timeout=400)
        
        if not raw:
            print("[EXTRACT] Risposta vuota dal task")
            return {}, []
        
        # MESSAGGIO DI CONFERMA
        print(f"[REGOLO OK] Risposta ricevuta (lunghezza={len(raw)} caratteri)")
        
        graph = parse_graph(raw)
        
        if not graph:
            print("[EXTRACT] Nessun grafo estratto (lista vuota)")
            return {}, []
        
    except Exception as e:
        print(f"[EXTRACT ERROR] Errore nel recuperare risultato task: {e}")
        import traceback
        traceback.print_exc()
        return {}, []

    # Merge identico al tuo
    nodes_chunk = {}
    relationships_chunk = []

    for entry in graph:
        node = entry.node
        target = entry.target_node
        rel = entry.relationship

        if node not in nodes_chunk:
            nodes_chunk[node] = str(uuid.uuid4())

        if target and target not in nodes_chunk:
            nodes_chunk[target] = str(uuid.uuid4())

        relationships_chunk.append({
            "node": node,
            "target_node": target,
            "relationship": rel
        })

    print(f"[EXTRACT] Estratti {len(nodes_chunk)} nodi e {len(relationships_chunk)} relazioni")
    return nodes_chunk, relationships_chunk

def build_graph_from_chunks(chunks, window_size: int = 4):
    """
    Costruisce il grafo da tutti i chunk usando Celery,
    ma tenendo al massimo `window_size` task in coda per volta.
    """
    print(f"[BUILD] Ho {len(chunks)} chunk, window={window_size}")

    all_nodes: dict[str, str] = {}
    all_relationships: list[dict] = []
    chunk_node_mapping: list[list[str]] = []

    total = len(chunks)

    for start in range(0, total, window_size):
        batch = chunks[start:start + window_size]
        print(f"[BUILD] Schedulo batch chunk {start}–{start + len(batch) - 1}")

        batch_tasks: list[tuple[int, object]] = []

        # 1) Schedulo solo il batch corrente
        for local_idx, chunk in enumerate(batch):
            idx = start + local_idx
            try:
                task = regolo_llm_parser(chunk)  # ritorna AsyncResult da regolo_call.delay(...)
                batch_tasks.append((idx, task))
                print(f"[BUILD] Task {idx+1}/{total} schedulato (id={task.id})")
            except Exception as e:
                print(f"[BUILD ERROR] Errore nella schedulazione chunk {idx}: {e}")

        # 2) Raccolgo i risultati del batch PRIMA di schedulare quelli dopo
        for idx, task in batch_tasks:
            try:
                nodes_chunk, rels_chunk = extract_graph_components(task)
                print(
                    f"[BUILD] Chunk {idx+1}/{total} processato: "
                    f"{len(nodes_chunk)} nodi, {len(rels_chunk)} relazioni"
                )
            except Exception as e:
                print(f"[BUILD ERROR] Errore nel processare chunk {idx}: {e}")
                nodes_chunk, rels_chunk = {}, []

            # --- MERGE GLOBALE (stesso schema di prima) ---
            temp_to_final: dict[str, str] = {}
            chunk_global_ids: list[str] = []

            for name, temp_id in nodes_chunk.items():
                if name not in all_nodes:
                    all_nodes[name] = temp_id
                temp_to_final[temp_id] = all_nodes[name]
                chunk_global_ids.append(all_nodes[name])

            if chunk_global_ids:
                chunk_node_mapping.append(chunk_global_ids)

            for rel in rels_chunk:
                src = rel["node"]
                tgt = rel["target_node"]
                relationship = rel["relationship"]

                src_id = temp_to_final[nodes_chunk[src]]
                tgt_id = temp_to_final[nodes_chunk[tgt]] if tgt else None

                all_relationships.append(
                    {"source": src_id, "target": tgt_id, "type": relationship}
                )

    return all_nodes, all_relationships, chunk_node_mapping



# =========================
# NEO4J INGEST
# =========================

def ingest_to_neo4j(nodes, relationships, driver=None):
    """
    Ingest nodes and relationships into Neo4j 
    Se driver è None, usa il neo4j_driver globale.
    """
    neo4j_client = driver or neo4j_driver

    with neo4j_client.session() as session:
        # Nodi: CREATE su id, aggiorna sempre il name
        for name, node_id in nodes.items():
            session.run(
                """
                CREATE (n:Entity {id: $id})
                SET n.name = $name
                """,
                id=node_id,
                name=name,
            )

        # Relazioni: CREATE per evitare duplicati
        for relationship in relationships:
            session.run(
                """
                MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
                CREATE (a)-[r:RELATIONSHIP {type: $type}]->(b)
                """,
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"],
            )

    return nodes

# =========================
# QDRANT COLLECTION & INGEST
# =========================

def create_collection(client, collection_name, vector_dimension):
    """
    Crea la collection Qdrant solo se non esiste.
    """
    try:
        client.get_collection(collection_name)
        print(f"La collection '{collection_name}' esiste già. Nessuna creazione necessaria.")
        return

    except Exception as e:
        # Caso previsto: collection inesistente
        if "Not found" in str(e):
            print(f"La collection '{collection_name}' non esiste. Creazione in corso...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE
                )
            )

            print(f"Collection '{collection_name}' creata correttamente.")
            return

        # Caso non previsto: errore reale
        print(f"Errore durante il controllo della collection: {e}")



def ingest_to_qdrant(
    collection_name: str,
    chunks: list[str],
    chunk_node_mapping: list[list[str]],
):
    """
    Ingestione in Qdrant:
    - chunks: lista di stringhe (testo per chunk)
    - chunk_node_mapping: per ogni chunk, lista di node_id globali associati

    Ogni punto in Qdrant rappresenta un chunk e conosce i nodi Neo4j a cui è collegato.
    """
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    points = []
    for idx, (emb, node_ids) in enumerate(zip(embeddings, chunk_node_mapping)):
        payload = {
            "text": chunks[idx],
            "nodes": node_ids,  # lista di id Neo4j collegati
        }

        # 👇 aggiungiamo un id "principale" per il retriever
        if node_ids:
            payload["id"] = node_ids[0]  # deve corrispondere a Entity.id in Neo4j

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload,
            )
        )


    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print(f"Inseriti {len(points)} punti in Qdrant.")
    return len(points)



def generate_query_embedding(query: str):
    emb = embedding_model.encode([query], convert_to_numpy=True)[0]
    return emb.tolist()


def retriever_search(
    neo4j_driver,
    qdrant_client,
    collection_name: str,
    query: str,
):
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    query_vector = generate_query_embedding(query)

    results = retriever.search(
        query_vector=query_vector,
        top_k=5,
    )

    return results


# =========================
# GRAPH CONTEXT & QA CON REGOLO
# =========================

def fetch_related_graph(neo4j_client, entity_ids):
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append(
                {
                    "entity": record["e"],
                    "relationship": record["r"],
                    "related_node": record["related"],
                }
            )
            if record["r2"] and record["n2"]:
                subgraph.append(
                    {
                        "entity": record["related"],
                        "relationship": record["r2"],
                        "related_node": record["n2"],
                    }
                )
    return subgraph


def format_graph_context(subgraph):
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}


def graphRAG_run(graph_context, user_query: str) -> str:
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
Sei un assistente che risponde usando SOLO il seguente grafo di conoscenza come contesto.

NODI:
{nodes_str}

ARCHI:
{edges_str}

Domanda dell'utente:
\"{user_query}\"

Rispondi in modo sintetico, preciso e aderente al grafo, in modo che l'utente possa comprendere la risposta.
"""
    task = regolo_call.delay(prompt)
    response = task.get(timeout=400)
    
    return response


def graphRAG_run_with_history(graph_context, user_query: str, conversation_history: Optional[List[dict]] = None) -> str:
    """
    Versione migliorata di graphRAG_run che può usare la storia della conversazione
    per dare risposte più contestuali.
    """
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    
    # Costruisci il contesto della conversazione se presente
    history_context = ""
    if conversation_history:
        history_context = "\n\nContesto della conversazione precedente:\n"
        for msg in conversation_history[-4:]:  # Ultimi 4 scambi (2 domande + 2 risposte)
            role = "Utente" if msg.get("role") == "user" else "Assistente"
            content = msg.get("content", "")
            history_context += f"{role}: {content}\n"
        history_context += "\nUsa questo contesto per dare risposte più coerenti e contestuali. Se la domanda fa riferimento a qualcosa detto prima, usa quel contesto.\n"
    
    prompt = f"""
Sei un assistente che risponde usando SOLO il seguente grafo di conoscenza come contesto.
{history_context}
NODI:
{nodes_str}

ARCHI:
{edges_str}

Domanda dell'utente:
"{user_query}"

Rispondi in modo sintetico, preciso e aderente al grafo, in modo che l'utente possa comprendere la risposta.
Se la domanda fa riferimento a qualcosa detto prima nella conversazione, usa quel contesto per dare una risposta più completa e coerente.
"""
    task = regolo_call.delay(prompt)
    response = task.get(timeout=400)
    
    return response


if __name__ == "__main__":
    start_time = time.perf_counter()
    print("Script started")
    print("Loading environment variables...")
    # (già fatto sopra con load_dotenv('.env.local'), qui solo log)
    print("Environment variables loaded")
    
    print("Initializing clients...")
    # (già inizializzati sopra, qui solo log)
    print("Clients initialized")
    
    print("Creating collection...")
    collection_name = "Bandi"
    vector_dimension = VECTOR_DIM
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")
    
    print("Extracting graph components...")


    doc_path = "DD_G13041_10_10_2025_Bando.pdf"


    print("Reading PDF and extracting text + tables...")
    raw_data = extract_pdf_text_with_tables(doc_path)

    print("Chunking raw_data...")
    chunks = chunk_text(raw_data, max_words=800)

    print(f"Chunks extracted: {len(chunks)}")
    if not chunks:
        print("No chunks extracted. Exiting.")
        sys.exit(0)

    # ================
    # 2) GRAFO DAI CHUNK
    # ================
    nodes, relationships, chunk_node_mapping = build_graph_from_chunks(chunks)

    print("Nodes:", nodes)
    print("Relationships:", relationships)
    
    # ================
    # 3) NEO4J
    # ================
    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")
    
    # ================
    # 4) QDRANT
    # ================
    print("Ingesting to Qdrant...")
    # ora passiamo la LISTA DI CHUNK, non la stringa intera
    ingest_to_qdrant(collection_name, chunks, chunk_node_mapping)
    print("Qdrant ingestion complete")

    # ================
    # 5) RETRIEVAL + GRAPHRAG
    # ================



    query = "Chi può applicare per questo bando?"



    print("Starting retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)
    
    print("Extracting entity IDs...")
    entity_ids = []
    for item in getattr(retriever_result, "items", []):
        try:
            entity_id = item.content.split("'id': '")[1].split("'")[0]
            entity_ids.append(entity_id)
        except Exception:
            continue
    print("Entity IDs:", entity_ids)
    
    if not entity_ids:
        print("No entity IDs found. Stopping before subgraph/GraphRAG.")
        sys.exit(0)

    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)
    
    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)
    
    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)
    end_time = time.perf_counter()
    print(f"[TIMER] Script finished in {end_time - start_time:.2f} seconds")
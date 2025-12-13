import os
import re
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
from celery import group

from typing import List, Optional,Any
from doctr.io import read_pdf
from doctr.models import ocr_predictor


class GraphEntry(BaseModel):
    node: str
    target_node: str | None
    relationship: str | None
    attributes: dict[str, Any] | None = None


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

qdrant_client = QdrantClient(
    url=qdrant_url,
    )



# Modello di embedding TESTO → VETTORE
# (puoi cambiarlo con un altro sentence transformer se vuoi)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM = embedding_model.get_sentence_embedding_dimension()


doctr_ocr = ocr_predictor(
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_vgg16_bn",
    pretrained=True,
)


# =========================
# PDF → TESTO (+ pseudo-tabelle)
# =========================

def build_internal_prompt(chunk_text: str) -> str:
    return f"""
Sei un estrattore estremamente preciso di relazioni per grafi di conoscenza
da bandi e documenti amministrativi in italiano.

DAL TESTO DEVI:
1. ESTRARRE IL MAGGIOR NUMERO POSSIBILE DI RELAZIONI CORRETTE TRA ENTITÀ.
2. ASSOCIARE AI NODI O ALLE RELAZIONI GLI ATTRIBUTI NUMERICI (importi in EUR, percentuali, durate, ecc.)
   quando sono chiaramente collegati.

Devi restituire ESCLUSIVAMENTE un JSON con la seguente struttura ESATTA:

{{
  "graph": [
    {{
      "node": "Nome entità di partenza",
      "target_node": "Nome entità collegata o null",
      "relationship": "Tipo di relazione o null",
      "attributes": {{
        "chiave_attributo_1": valore_numerico_o_testuale,
        "chiave_attributo_2": valore_numerico_o_testuale
      }}
    }}
  ]
}}

REGOLE:
- Usa ESATTAMENTE le chiavi: "graph", "node", "target_node", "relationship", "attributes".
- Se non ci sono attributi per quella relazione, usa "attributes": {{}}
- Gli attributi numerici devono essere NUMERI (non stringhe), es:
  - "contributo_massimo_eur": 50000.0
  - "intensita_aiuto_pct": 0.50
- NON creare nodi separati per numeri puri (es. "50%" non deve diventare un nodo).
- I numeri già normalizzati nel testo (es. "EUR_50000.00", "PCT_0.5000") vanno mappati in attributi numerici.

ESEMPI DI BUON ATTRIBUTO:
- un articolo che definisce il contributo massimo: 
  node = "Contributo massimo"
  target_node = "Intervento agevolato"
  relationship = "LIMITA"
  attributes = {{"contributo_massimo_eur": 50000.0}}

- una percentuale di cofinanziamento:
  node = "Cofinanziamento"
  target_node = "Beneficiari"
  relationship = "RICHIEDE"
  attributes = {{"cofinanziamento_min_pct": 0.20}}

SE NON CI SONO RELAZIONI, restituisci:
{{ "graph": [] }}

Testo da analizzare:
\"\"\"{chunk_text}\"\"\"
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

def is_scanned_page(page: fitz.Page, text_min_chars: int = 80) -> bool:
    """
    Ritorna True se la pagina sembra scansionata:
    - pochissimo testo "nativo"
    - presenza di immagini
    """
    text = page.get_text("text") or ""
    has_images = len(page.get_images(full=True)) > 0
    return (len(text.strip()) < text_min_chars) and has_images


def doctr_ocr_per_page(pdf_path: str) -> list[str]:
    """
    Usa DocTR per fare OCR dell'intero PDF e ritorna una lista di testi per pagina.
    """
    doc = read_pdf(pdf_path)
    result = doctr_ocr(doc)
    # result.export() -> dict annidato: pages -> blocks -> lines -> words
    exported = result.export()
    page_texts: list[str] = []

    for page in exported["pages"]:
        lines_out = []
        for block in page["blocks"]:
            for line in block["lines"]:
                words = [w["value"] for w in line["words"]]
                lines_out.append(" ".join(words))
        page_texts.append("\n".join(lines_out))

    return page_texts

def extract_pdf_text_with_tables(pdf_path: str) -> str:
    """
    Estrae testo da PDF:
    - usa PyMuPDF per il testo nativo + pseudo-tabelle
    - usa DocTR per le pagine scansionate (fallback)
    Ritorna un unico testo pronto per chunking.
    """
    doc = fitz.open(pdf_path)
    all_page_texts: list[str] = []

    # 1) Pre-scan: individua pagine scansionate
    scanned_flags = [is_scanned_page(doc[i]) for i in range(doc.page_count)]

    # 2) Se serve, prepara OCR DocTR una sola volta
    doctr_pages: list[str] | None = None
    if any(scanned_flags):
        print(f"[OCR] Rilevate {sum(scanned_flags)} pagine scansionate, uso DocTR...")
        doctr_pages = doctr_ocr_per_page(pdf_path)

    for page_index in range(doc.page_count):
        page = doc[page_index]

        # testo base
        page_text = page.get_text("text") or ""

        # se pagina scansionata, sovrascrivo con OCR
        if scanned_flags[page_index] and doctr_pages is not None:
            ocr_text = doctr_pages[page_index]
            if ocr_text and ocr_text.strip():
                page_text = ocr_text

        # pseudo-tabelle da PyMuPDF (utile anche su OCR)
        blocks = page.get_text("blocks")
        table_texts = []

        for b in blocks:
            text_block = b[4]
            if "|" in text_block or "\t" in text_block:
                cleaned = text_block.replace("\t", " | ")
                table_texts.append(cleaned)

        if table_texts:
            tables_serialized = "\n[TABELLA]:\n" + "\n".join(table_texts)
            full_page_text = page_text + "\n" + tables_serialized
        else:
            full_page_text = page_text

        all_page_texts.append(full_page_text)

    doc.close()

    full_text = "\n\n".join(all_page_texts)
    return full_text

def normalize_whitespace(text: str) -> str:
    # rimuovi spazi doppi
    text = re.sub(r"[ \t]+", " ", text)
    # normalizza righe vuote multiple
    text = re.sub(r"\n{3,}", "\n\n", text)
    # trim globale
    return text.strip()


def chunk_text(
    text: str,
    max_words: int = 800,
    overlap_words: int = 80,
) -> list[str]:
    """
    Chunking word-based con overlap per conservare contesto locale.
    """
    words = text.split()
    n = len(words)
    chunks: list[str] = []

    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + max_words, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        # avanzamento con overlap
        start = end - overlap_words

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
        attrs = entry.attributes or {}

        if node not in nodes_chunk:
            nodes_chunk[node] = str(uuid.uuid4())

        if target and target not in nodes_chunk:
            nodes_chunk[target] = str(uuid.uuid4())

        relationships_chunk.append({
            "node": node,
            "target_node": target,
            "relationship": rel,
            "attributes": attrs,
        })

    print(f"[EXTRACT] Estratti {len(nodes_chunk)} nodi e {len(relationships_chunk)} relazioni")
    return nodes_chunk, relationships_chunk

def build_graph_from_chunks(chunks, window_size: int = 8):
    """
    Costruisce il grafo da tutti i chunk usando Celery groups
    per eseguire fino a `window_size` task simultaneamente.
    """
    print(f"[BUILD] Ho {len(chunks)} chunk, concurrency={window_size}")

    all_nodes: dict[str, str] = {}
    all_relationships: list[dict] = []
    chunk_node_mapping: list[list[str]] = []

    total = len(chunks)

    # Dividi i chunk in gruppi di window_size
    for start in range(0, total, window_size):
        batch = chunks[start:start + window_size]
        batch_indices = list(range(start, start + len(batch)))
        print(f"[BUILD] Schedulo gruppo chunk {start}–{start + len(batch) - 1}")

        # Crea un gruppo Celery con tutti i task del batch
        # Ogni task chiama regolo_llm_parser che ritorna un AsyncResult
        # Ma per i gruppi, dobbiamo passare direttamente la funzione task
        group_tasks = []
        for idx, chunk in zip(batch_indices, batch):
            # Crea il task usando regolo_call.s() per signature
            internal_prompt = build_internal_prompt(chunk)
            group_tasks.append(regolo_call.s(internal_prompt))
            print(f"[BUILD] Task {idx+1}/{total} aggiunto al gruppo")

        # Crea e esegui il gruppo Celery
        job = group(group_tasks)
        print(f"[BUILD] Esecuzione gruppo con {len(group_tasks)} task in parallelo...")
        result_group = job.apply_async()

        # Attendi che tutti i task del gruppo completino e raccogli i risultati
        try:
            # get() su un GroupResult ritorna una lista di risultati nell'ordine dei task
            raw_results = result_group.get(timeout=400)
            
            # Processa ogni risultato del gruppo
            for idx, raw in zip(batch_indices, raw_results):
                try:
                    # Crea un task fittizio per usare extract_graph_components
                    # che si aspetta un task con .get()
                    class MockTask:
                        def __init__(self, result):
                            self._result = result
                        def get(self, timeout=None):
                            return self._result
                    
                    mock_task = MockTask(raw)
                    nodes_chunk, rels_chunk = extract_graph_components(mock_task)
                    print(
                        f"[BUILD] Chunk {idx+1}/{total} processato: "
                        f"{len(nodes_chunk)} nodi, {len(rels_chunk)} relazioni"
                    )
                except Exception as e:
                    print(f"[BUILD ERROR] Errore nel processare chunk {idx}: {e}")
                    import traceback
                    traceback.print_exc()
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
                    attrs = rel.get("attributes", {}) or {}

                    src_id = temp_to_final[nodes_chunk[src]]

                    tgt_id = None
                    if tgt:
                        tgt_id = temp_to_final[nodes_chunk[tgt]]

                    all_relationships.append(
                        {
                            "source": src_id,
                            "target": tgt_id,
                            "type": relationship,
                            "attributes": attrs,
                        }
                    )

        except Exception as e:
            print(f"[BUILD ERROR] Errore nell'esecuzione del gruppo: {e}")
            import traceback
            traceback.print_exc()

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
            SET r += $attributes
            """,
            source_id=relationship["source"],
            target_id=relationship["target"],
            type=relationship["type"],
            attributes=relationship.get("attributes", {}),
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
import os
import requests
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



REGOLO_API_URL = "https://api.openai.com/v1/chat/completions"
REGOLO_KEY = os.getenv("OPENAI_API_KEY")

# ⚠️ USA UN MODELLO VALIDO
REGOLO_MODEL = "gpt-4.1-mini"

#Env e Clients
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_client = QdrantClient(url=qdrant_url)   #Qdrant
qdrant_key = os.getenv("QDRANT_KEY")           

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # Modelli di embedding
VECTOR_DIM = embedding_model.get_sentence_embedding_dimension() 

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")      #Neo4j
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# OCR per documenti scansionati



def generate_query_embedding(query: str) -> List[float]:
    """Genera embedding per la query."""
    emb = embedding_model.encode([query], convert_to_numpy=True)[0]
    return emb.tolist()

def retrieve_graph_context(
    query: str,
    collection_name: str = "Bandi",
    top_k: int = 10
) -> Tuple[List[str], List[str], Dict]:
    """
    Recupera contesto per il RAG:
    1. Cerca in Qdrant i chunk più rilevanti
    2. Estrae i nodi collegati da quei chunk
    3. Recupera il subgraph Neo4j per quei nodi
    """
    # 1. Ricerca semantica in Qdrant
    query_vector = generate_query_embedding(query)
    
    # Correzione: usa limit direttamente senza moltiplicare
    search_results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_vector,  # Nota: il parametro si chiama ora 'query'
    limit=top_k,  # ← QUI: stai chiedendo 5 volte i chunk necessari
        )  # ← Questo potrebbe non essere corretto
    
    # Estrai i punti correttamente
    points = search_results.points
    
    # 2. Estrai nodi dai risultati Qdrant
    all_node_ids = set()
    qdrant_texts = []
    
    for point in points:
        if point.payload:
            # Testo del chunk
            if "text" in point.payload:
                qdrant_texts.append(point.payload["text"])
            
            # Nodi collegati
            if "nodes" in point.payload:
                for node_id in point.payload["nodes"]:
                    all_node_ids.add(node_id)
    
    
    if not all_node_ids:
        print("[RETRIEVAL] Nessun nodo trovato in Qdrant")
        return [], [], {}
    
    # 3. Recupera subgraph Neo4j
    entity_ids = list(all_node_ids)
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    
    # 4. Formatta il contesto
    graph_context = format_graph_context(subgraph, qdrant_texts)
    
    return entity_ids, qdrant_texts, graph_context


#Graph Query

def fetch_related_graph(neo4j_client, entity_ids: List[str], max_relationships: int = 100) -> List[Dict]:
    """
    Recupera il subgraph con tutti gli attributi dei nodi.
    LIMITATO a max_relationships per evitare overflow.
    """
    if not entity_ids:
        return []
    
    query = """
    MATCH (e)
    WHERE e.id IN $entity_ids
    WITH e, COLLECT(e) as sources
    OPTIONAL MATCH (e)-[r1]-(n1)
    WHERE type(r1) IS NOT NULL
    WITH sources, e, r1, n1, COLLECT({e: e, r1: r1, n1: n1}) as first_hop
    OPTIONAL MATCH (n1)-[r2]-(n2)
    WHERE type(r2) IS NOT NULL AND n2.id IS NOT NULL
    RETURN 
      e as source_node,
      e.id as source_id,
      e.name as source_name,
      labels(e) as source_labels,
      type(r1) as rel1_type,
      properties(r1) as rel1_props,
      n1 as target1_node,
      n1.id as target1_id,
      n1.name as target1_name,
      labels(n1) as target1_labels,
      type(r2) as rel2_type,
      properties(r2) as rel2_props,
      n2 as target2_node,
      n2.id as target2_id,
      n2.name as target2_name,
      labels(n2) as target2_labels
    LIMIT $limit
    """
    
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids, limit=max_relationships)
        subgraph = []
        processed_rels = set()
        
        for record in result:
            # Verifica che il nodo sorgente esista
            if not record["source_node"]:
                continue
                
            # Aggiungi relazione 1 se esiste
            if record["rel1_type"] and record["target1_node"]:
                rel_key = f"{record['source_id']}|{record['target1_id']}|{record['rel1_type']}"
                if rel_key not in processed_rels:
                    subgraph.append({
                        "source": {
                            "id": record["source_id"],
                            "name": record["source_name"],
                            "labels": record["source_labels"],
                            "attributes": dict(record["source_node"])
                        },
                        "target": {
                            "id": record["target1_id"],
                            "name": record["target1_name"],
                            "labels": record["target1_labels"],
                            "attributes": dict(record["target1_node"])
                        },
                        "relationship": {
                            "type": record["rel1_type"],
                            "attributes": dict(record["rel1_props"]) if record["rel1_props"] else {}
                        }
                    })
                    processed_rels.add(rel_key)
            
            # Aggiungi relazione 2 se esiste
            if record["rel2_type"] and record["target2_node"]:
                rel_key = f"{record['target1_id']}|{record['target2_id']}|{record['rel2_type']}"
                if rel_key not in processed_rels:
                    subgraph.append({
                        "source": {
                            "id": record["target1_id"],
                            "name": record["target1_name"],
                            "labels": record["target1_labels"],
                            "attributes": dict(record["target1_node"])
                        },
                        "target": {
                            "id": record["target2_id"],
                            "name": record["target2_name"],
                            "labels": record["target2_labels"],
                            "attributes": dict(record["target2_node"])
                        },
                        "relationship": {
                            "type": record["rel2_type"],
                            "attributes": dict(record["rel2_props"]) if record["rel2_props"] else {}
                        }
                    })
                    processed_rels.add(rel_key)
    
    print(f"[DEBUG] Subgraph recuperato con {len(subgraph)} relazioni (limite: {max_relationships})")
    return subgraph

def format_graph_context(subgraph: List[Dict], qdrant_texts: List[str]) -> Dict:
    """Formatta il contesto del grafo per il LLM."""
    nodes_info = {}
    edges_info = []
    
    for entry in subgraph:
        source = entry["source"]
        target = entry["target"]
        rel = entry["relationship"]
        
        # Memorizza informazioni sui nodi
        if source["id"] not in nodes_info:
            nodes_info[source["id"]] = {
                "name": source["name"],
                "labels": source.get("labels", []),
                "attributes": source.get("attributes", {})
            }
        
        if target["id"] not in nodes_info:
            nodes_info[target["id"]] = {
                "name": target["name"],
                "labels": target.get("labels", []),
                "attributes": target.get("attributes", {})
            }
        
        # Formatta l'arco
        edge_info = {
            "from": source["name"],
            "to": target["name"],
            "type": rel["type"],
            "attributes": rel.get("attributes", {})
        }
        edges_info.append(edge_info)
    
    # Formatta i nodi in modo strutturato
    nodes_formatted = []
    for node_id, info in nodes_info.items():
        name = info["name"]
        labels = info.get("labels", [])
        attrs = info.get("attributes", {})
        
        # Filtra attributi standard
        filtered_attrs = {}
        for k, v in attrs.items():
            if k.lower() not in ["id", "name", "label", "__properties__"]:
                filtered_attrs[k] = v
        
        # Costruisci stringa formattata
        node_str = f"{name}"
        if labels:
            node_str += f" [{', '.join(labels)}]"
        
        if filtered_attrs:
            attrs_lines = []
            for attr_key, attr_value in filtered_attrs.items():
                attrs_lines.append(f"    - {attr_key}: {attr_value}")
            node_str += ":\n" + "\n".join(attrs_lines)
        
        nodes_formatted.append(node_str)
    
    # Formatta i testi Qdrant
    qdrant_context = "\n\n".join([
        f"[Riferimento {i+1}]:\n{text}"
        for i, text in enumerate(qdrant_texts[:10])  # Limita a 3 per brevità
    ]) if qdrant_texts else "Nessun riferimento testuale trovato."
    
    return {
        "nodes": nodes_formatted,
        "edges": edges_info,
        "qdrant_context": qdrant_context,
        "nodes_info": nodes_info,
        "edges_info": edges_info
    }


HEADERS = {
    "Authorization": f"Bearer {REGOLO_KEY}",
    "Content-Type": "application/json",
}

def llm_call(prompt: str, response_format: str = "text") -> str:
    """
    Chiama OpenAI Chat Completions
    response_format: "text" | "json"
    """

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # JSON mode (supportato da gpt-4o / gpt-4.1 / mini)
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    print(f"[WORKER] Regolo call start")

    try:
        resp = requests.post(
            REGOLO_API_URL,
            headers=HEADERS,
            json=payload,
            timeout=None, 
        )

        if resp.status_code >= 400:
            print("[WORKER][OPENAI ERROR]")
            print(resp.status_code)
            print(resp.text)

        resp.raise_for_status()
        data = resp.json()

        message = (
            data
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return message

    except Exception as e:
        print(f"[WORKER][EXCEPTION] {e}")
        raise e

def graphRAG_run_with_history(graph_context, user_query: str, conversation_history: Optional[List[dict]] = None) -> str:
    """
    Versione migliorata di graphRAG_run che può usare la storia della conversazione
    per dare risposte più contestuali, con formattazione strutturata dei nodi.
    """
    nodes_str = "\n".join(graph_context["nodes"])
    
    # Correzione: formatta gli edges (che sono dizionari) in stringhe
    edges_info = graph_context.get("edges", [])
    edges_list = []
    
    for edge in edges_info:
        # edge è un dizionario con: from, to, type, attributes
        edge_str = f"{edge.get('from', '?')} --[{edge.get('type', '?')}]--> {edge.get('to', '?')}"
        
        # Aggiungi attributi se presenti
        attrs = edge.get('attributes', {})
        if attrs:
            attrs_str = ", ".join([f"{k}: {v}" for k, v in attrs.items()])
            edge_str += f" ({attrs_str})"
        
        edges_list.append(edge_str)
    
    edges_str = "; ".join(edges_list)
    
    qdrant_context = graph_context.get("qdrant_context", "")
    print("Nodi:", nodes_str[:500] + "..." if len(nodes_str) > 500 else nodes_str)
    print("Archi:", edges_str[:500] + "..." if len(edges_str) > 500 else edges_str)
    print("Contesto Qdrant:", qdrant_context[:1000000000] + "..." if len(qdrant_context) > 100000000 else qdrant_context)
    
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
NODI (ogni nodo ha i suoi attributi elencati sotto, usa questi dettagli nella risposta):
{nodes_str}

ARCHI (relazioni tra nodi):
{edges_str}

RIFERIMENTI DAL DOCUMENTO:
{qdrant_context}

Domanda dell'utente:
"{user_query}"

Analizza attentamente:
1. Gli attributi di ogni nodo (informazioni sotto ogni nome di nodo)
2. Le relazioni tra i nodi (archi)
3. Il contesto della conversazione precedente (se presente)
4. I riferimenti dal documento

Rispondi in modo sintetico, preciso e aderente al grafo:
- CITANDO ESPLICITAMENTE GLI ATTRIBUTI DEI NODI quando rilevanti
- MENZIONANDO LE RELAZIONI tra i nodi quando importanti
- USA I VALORI SPECIFICI degli attributi se la domanda li richiede
- RIFERISCITI AL CONTESTO della conversazione se appropriato

La risposta deve essere informativa ma concisa.
"""
    response = llm_call(prompt)
    
    return response

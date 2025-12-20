import os
import re
import sys
import time, datetime
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime

import fitz  # PyMuPDF
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient, models
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from celery_app import regolo_call
from celery import group

from doctr.io import read_pdf
from doctr.models import ocr_predictor

# =========================
# MODELLI PYDANTIC
# =========================

class GraphEntry(BaseModel):
    node: str
    target_node: str | None
    relationship: str | None
    attributes: dict[str, Any] | None = None


class GraphComponents(BaseModel):
    graph: list[GraphEntry]


class BandoOntology(BaseModel):
    """Ontologia strutturata per un bando basata sulla tua analisi"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # A. Avviso
    identificativo: str
    autorita: str  # "Regione Lazio"
    soggetto_attuatore: Optional[str] = None  # "Lazio Innova"
    dotazione_finanziaria: Optional[float] = None  # 15000000.0
    regime_aiuto: Optional[str] = None  # "De Minimis"
    
    # Finestre temporali
    data_apertura_formulario: Optional[str] = None  # "2025-10-16T12:00:00"
    data_inizio_domande: Optional[str] = None  # "2025-10-24T12:00:00"
    data_fine_domande: Optional[str] = None  # "2025-11-24T17:00:00"
    
    piattaforma_presentazione: Optional[str] = None  # "GeCoWEB Plus"
    
    # B. Beneficiario
    dimensioni_ammesse: List[str] = []  # ["Micro", "Piccola", "Media"]
    requisiti_beneficiario: List[str] = []  # ["iscrizione RI", "sede Lazio", "DURC regolare"]
    sede_obbligatoria: Optional[str] = None  # "Lazio"
    
    # Attributi bonus
    attributi_bonus: List[str] = []  # ["impresa_giovanile", "certificazione_parita_genere", "certificazioni_sostenibilita"]
    
    # C. Progetto
    tipologie_intervento: List[str] = []  # ["A", "B", "C", "D", "E"]
    sottotipologie_cloud: List[str] = []  # ["D.1", "D.2", "D.3", "D.4"]
    importo_minimo_progetto: Optional[float] = None  # 14000.0
    tempo_realizzazione_mesi: Optional[int] = None  # 6
    
    # Esclusioni specifiche
    esclusioni: List[str] = []  # ["già beneficiari 2023/2024", "interventi identici già agevolati"]
    
    # D. Intervento (dettagli per tipologia)
    interventi: List[dict] = []  # [{"tipo": "A", "nome": "Diagnosi Digitale", ...}]
    
    # E. Contributo
    natura_contributo: Optional[str] = None  # "fondo perduto"
    importi_unitari: dict = {}  # {"A": {"Micro": 0, "Piccola": 8162.40, ...}}
    massimali_dimensione: dict = {}  # {"Micro": 50000, "Piccola": 100000, "Media": 150000}
    incompatibilita_aiuti: bool = False
    
    # F. Criteri di Selezione
    criteri_selezione: List[dict] = []  # [{"codice": "C1", "nome": "valore_aggiunto_per_addetto", "max_punti": 35, ...}]
    criterio_tiebreak: Optional[str] = None
    
    # G. Procedura Domanda
    allegati_obbligatori: List[str] = []
    piattaforma_invio: Optional[str] = None
    regola_senza_soccorso: bool = False
    
    # H. Istruttoria
    ordine_istruttoria: Optional[str] = None  # "basato sul punteggio"
    
    # I. Erogazione
    modalita_erogazione: Optional[str] = None
    documenti_erogazione: List[str] = []
    
    # L. Revoca/Rinuncia
    condizioni_revoca: List[str] = []
    
    # Metadata
    file_name: Optional[str] = None
    pagine_totali: Optional[int] = None
    data_processing: str = Field(default_factory=lambda: datetime.now().isoformat())
    fonte: Optional[str] = None  # "Allegato 1 all'atto G13041/10.10.2025"

# =========================
# ENV & CLIENTS
# =========================

load_dotenv()

qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
qdrant_client = QdrantClient(url=qdrant_url)

# Modelli di embedding
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM = embedding_model.get_sentence_embedding_dimension()

# OCR per documenti scansionati
doctr_ocr = ocr_predictor(
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_vgg16_bn",
    pretrained=True,
)

# =========================
# FUNZIONI DI ESTRAZIONE PDF
# =========================

def is_scanned_page(page: fitz.Page, text_min_chars: int = 80) -> bool:
    """Ritorna True se la pagina sembra scansionata."""
    text = page.get_text("text") or ""
    has_images = len(page.get_images(full=True)) > 0
    return (len(text.strip()) < text_min_chars) and has_images

def doctr_ocr_per_page(pdf_path: str) -> list[str]:
    """Usa DocTR per fare OCR dell'intero PDF."""
    doc = read_pdf(pdf_path)
    result = doctr_ocr(doc)
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
    """Estrae testo da PDF con fallback OCR per pagine scansionate."""
    doc = fitz.open(pdf_path)
    all_page_texts: list[str] = []
    
    # Pre-scan: individua pagine scansionate
    scanned_flags = [is_scanned_page(doc[i]) for i in range(doc.page_count)]
    
    # Preparazione OCR se necessario
    doctr_pages: list[str] | None = None
    if any(scanned_flags):
        print(f"[OCR] Rilevate {sum(scanned_flags)} pagine scansionate, uso DocTR...")
        doctr_pages = doctr_ocr_per_page(pdf_path)
    
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_text = page.get_text("text") or ""
        
        # Sovrascrivi con OCR se pagina scansionata
        if scanned_flags[page_index] and doctr_pages is not None:
            ocr_text = doctr_pages[page_index]
            if ocr_text and ocr_text.strip():
                page_text = ocr_text
        
        # Pseudo-tabelle
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
    """Normalizza spazi e righe vuote."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_words: int = 800, overlap_words: int = 80) -> list[str]:
    """Chunking word-based con overlap."""
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
        start = end - overlap_words
    
    return chunks

# =========================
# PROMPT PER ONTOLOGIA COMPLETA
# =========================

def build_ontology_prompt_complete(full_text: str) -> str:
    """Crea il prompt per estrarre l'ontologia completa dal documento."""
    return f"""
Sei un esperto analista di bandi di finanziamento. 
Analizza il seguente bando e estrai TUTTE le informazioni strutturate seguendo lo schema JSON sottostante.

**ISTRUZIONI CRITICHE:**
1. Estratti SOLO informazioni presenti nel testo. Non inventare nulla.
2. Per le date, converti in formato ISO 8601 se possibile: "16/10/2025 h 12:00" → "2025-10-16T12:00:00"
3. Per gli importi, rimuovi '€' e converti in numero float: "€ 15.000.000" → 15000000.0
4. Per le liste, includi tutti gli elementi rilevanti trovati nel testo.
5. Se un campo non è presente, lascia null o lista vuota.
6. Raccogli TUTTI i dettagli strutturati, inclusi:
   - Tutti i criteri di selezione (C1-C7)
   - Tutti gli importi per ogni tipologia di intervento
   - Tutti i requisiti del beneficiario
   - Tutti gli interventi con le loro descrizioni

**STRUTTURA JSON OBBLIGATORIA:**
{{
  "identificativo": "string (es. 'Voucher Digitalizzazione PMI – II Edizione 2025')",
  "autorita": "string (es. 'Regione Lazio')",
  "soggetto_attuatore": "string | null (es. 'Lazio Innova')",
  "dotazione_finanziaria": "number | null",
  "regime_aiuto": "string | null (es. 'De Minimis')",
  "data_apertura_formulario": "string | null (ISO 8601)",
  "data_inizio_domande": "string | null (ISO 8601)",
  "data_fine_domande": "string | null (ISO 8601)",
  "piattaforma_presentazione": "string | null",
  "dimensioni_ammesse": ["lista di stringhe"],
  "requisiti_beneficiario": ["lista di stringhe"],
  "sede_obbligatoria": "string | null",
  "attributi_bonus": ["lista di stringhe"],
  "tipologie_intervento": ["lista di stringhe"],
  "importo_minimo_progetto": "number | null",
  "tempo_realizzazione_mesi": "number | null",
  "esclusioni": ["lista di stringhe"],
  "interventi": [
    {{
      "tipo": "string (es. 'A')",
      "nome": "string",
      "descrizione": "string",
      "restrizioni": ["lista di stringhe | null"]
    }}
  ],
  "natura_contributo": "string | null",
  "importi_unitari": {{}},
  "massimali_dimensione": {{}},
  "incompatibilita_aiuti": "boolean",
  "criteri_selezione": [
    {{
      "codice": "string",
      "nome": "string",
      "max_punti": "number",
      "descrizione": "string",
      "formula": "string | null"
    }}
  ],
  "criterio_tiebreak": "string | null",
  "allegati_obbligatori": ["lista di stringhe"],
  "piattaforma_invio": "string | null",
  "regola_senza_soccorso": "boolean",
  "ordine_istruttoria": "string | null",
  "modalita_erogazione": "string | null",
  "documenti_erogazione": ["lista di stringhe"],
  "condizioni_revoca": ["lista di stringhe"]
}}

**Testo del bando:**
\"\"\"{full_text}\"\"\"

Rispondi SOLO con il JSON. Non includere testo aggiuntivo.
"""

def extract_ontology_from_text(full_text: str) -> BandoOntology:
    """Estrae l'ontologia completa dal testo usando una singola chiamata LLM."""
    prompt = build_ontology_prompt_complete(full_text)
    task = regolo_call.delay(prompt)
    response = task.get(timeout=400)
    
    if not response:
        raise ValueError("Risposta vuota dal LLM")
    
    # Estrai JSON dalla risposta
    json_str = response.strip()
    if not json_str.startswith("{"):
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx+1]
        else:
            raise ValueError(f"JSON non trovato nella risposta: {response[:500]}")
    
    try:
        data = json.loads(json_str)
        # Aggiungi metadata mancanti
        data["id"] = str(uuid.uuid4())
        data["data_processing"] = datetime.now().isoformat()
        return BandoOntology(**data)
    except Exception as e:
        print(f"[ERROR] Errore nel parsing JSON: {e}")
        print(f"[ERROR] JSON ricevuto: {json_str[:1000]}")
        raise

# =========================
# CREAZIONE GRAFO ONTOLOGICO
# =========================

def create_ontology_graph(ontology: BandoOntology) -> Tuple[Dict[str, str], List[Dict], Dict]:
    """
    Crea un grafo strutturato dall'ontologia del bando.
    
    Returns:
        Tuple[nodes_dict, relationships_list, bando_attrs_dict]
        nodes_dict: mappatura nome_nodo -> id_univoco
        relationships_list: lista di relazioni con source, target, type, attributes
        bando_attrs_dict: attributi del nodo bando principale
    """
    nodes: Dict[str, str] = {}
    relationships: List[Dict] = []
    
    # 1. NODO PRINCIPALE DEL BANDO
    bando_id = ontology.id
    bando_name = f"BANDO: {ontology.identificativo}"
    nodes[bando_name] = bando_id
    
    # Attributi del bando come proprietà del nodo
    bando_attrs = {
        "identificativo": ontology.identificativo,
        "autorita": ontology.autorita,
        "dotazione_finanziaria": ontology.dotazione_finanziaria,
        "regime_aiuto": ontology.regime_aiuto,
        "data_fine_domande": ontology.data_fine_domande,
        "piattaforma_presentazione": ontology.piattaforma_presentazione,
        "importo_minimo_progetto": ontology.importo_minimo_progetto,
        "tempo_realizzazione_mesi": ontology.tempo_realizzazione_mesi,
        "file_name": ontology.file_name
    }
    # Filtra None
    bando_attrs = {k: v for k, v in bando_attrs.items() if v is not None}
    
    # 2. NODI PER DIMENSIONI AMMESSE
    for dim in ontology.dimensioni_ammesse:
        dim_name = f"DIMENSIONE: {dim}"
        dim_id = str(uuid.uuid4())
        nodes[dim_name] = dim_id
        relationships.append({
            "source": bando_id,
            "target": dim_id,
            "type": "PERMESSO_PER",
            "attributes": {"dimensione": dim}
        })
    
    # 3. NODI PER TIPOLOGIE DI INTERVENTO
    for intervento in ontology.interventi:
        tipo = intervento.get("tipo", "")
        nome = intervento.get("nome", "")
        
        if tipo and nome:
            intervento_name = f"INTERVENTO {tipo}: {nome}"
            intervento_id = str(uuid.uuid4())
            nodes[intervento_name] = intervento_id
            
            # Attributi dell'intervento
            intervento_attrs = {
                "tipo": tipo,
                "nome": nome,
                "descrizione": intervento.get("descrizione"),
                "restrizioni": intervento.get("restrizioni")
            }
            intervento_attrs = {k: v for k, v in intervento_attrs.items() if v is not None}
            
            relationships.append({
                "source": bando_id,
                "target": intervento_id,
                "type": "FINANZIA_INTERVENTO",
                "attributes": intervento_attrs
            })
            
            # COLGAMENTO IMPORTI - CORREZIONE QUI
            # Controlla se esiste il dizionario per questa tipologia
            if tipo in ontology.importi_unitari:
                importi = ontology.importi_unitari[tipo]
                
                # Se importi è un dizionario (tipologia -> dimensione -> importo)
                if isinstance(importi, dict):
                    for dimensione, importo in importi.items():
                        if importo and importo > 0:
                            importo_name = f"IMPORTO {tipo} {dimensione}: €{importo}"
                            importo_id = str(uuid.uuid4())
                            nodes[importo_name] = importo_id
                            relationships.append({
                                "source": intervento_id,
                                "target": importo_id,
                                "type": "HA_COSTO",
                                "attributes": {"importo": importo, "dimensione": dimensione}
                            })
                # Se importi è un numero float/direct value (solo importo fisso)
                elif isinstance(importi, (int, float)) and importi > 0:
                    importo_name = f"IMPORTO {tipo}: €{importi}"
                    importo_id = str(uuid.uuid4())
                    nodes[importo_name] = importo_id
                    relationships.append({
                        "source": intervento_id,
                        "target": importo_id,
                        "type": "HA_COSTO",
                        "attributes": {"importo": importi, "dimensione": "tutte"}
                    })
    
    # 4. NODI PER CRITERI DI SELEZIONE
    for criterio in ontology.criteri_selezione:
        codice = criterio.get("codice", "")
        nome = criterio.get("nome", "")
        
        if codice and nome:
            criterio_name = f"CRITERIO {codice}: {nome}"
            criterio_id = str(uuid.uuid4())
            nodes[criterio_name] = criterio_id
            
            criterio_attrs = {
                "codice": codice,
                "nome": nome,
                "max_punti": criterio.get("max_punti"),
                "formula": criterio.get("formula"),
                "descrizione": criterio.get("descrizione")
            }
            criterio_attrs = {k: v for k, v in criterio_attrs.items() if v is not None}
            
            relationships.append({
                "source": bando_id,
                "target": criterio_id,
                "type": "USA_CRITERIO",
                "attributes": criterio_attrs
            })
    
    # 5. NODI PER REQUISITI
    for req in ontology.requisiti_beneficiario:
        req_name = f"REQUISITO: {req}"
        req_id = str(uuid.uuid4())
        nodes[req_name] = req_id
        relationships.append({
            "source": bando_id,
            "target": req_id,
            "type": "RICHIDE",
            "attributes": {"requisito": req}
        })
    
    # 6. NODO LOCALIZZAZIONE
    if ontology.sede_obbligatoria:
        loc_name = f"LOCALIZZAZIONE: {ontology.sede_obbligatoria}"
        loc_id = str(uuid.uuid4())
        nodes[loc_name] = loc_id
        relationships.append({
            "source": bando_id,
            "target": loc_id,
            "type": "LIMITATO_A",
            "attributes": {"area": ontology.sede_obbligatoria}
        })
    
    # 7. NODI PER MASSIMALI - CORREZIONE QUI
    if isinstance(ontology.massimali_dimensione, dict):
        for dimensione, importo in ontology.massimali_dimensione.items():
            if importo:
                massimale_name = f"MASSIMALE {dimensione}: €{importo}"
                massimale_id = str(uuid.uuid4())
                nodes[massimale_name] = massimale_id
                relationships.append({
                    "source": bando_id,
                    "target": massimale_id,
                    "type": "IMPONE_LIMITE",
                    "attributes": {"importo_max": importo, "dimensione": dimensione}
                })
    
    # 8. NODI PER ATTRIBUTI BONUS
    for attributo in ontology.attributi_bonus:
        attributo_name = f"ATTRIBUTO_BONUS: {attributo}"
        attributo_id = str(uuid.uuid4())
        nodes[attributo_name] = attributo_id
        relationships.append({
            "source": bando_id,
            "target": attributo_id,
            "type": "RICONOSCE_BONUS",
            "attributes": {"attributo": attributo}
        })
    
    print(f"[ONTOLOGY GRAPH] Creati {len(nodes)} nodi e {len(relationships)} relazioni")
    return nodes, relationships, bando_attrs

# =========================
# INGESTIONE IN NEO4J
# =========================

def ingest_to_neo4j(nodes: Dict[str, str], relationships: List[Dict], driver=None, bando_attrs: Dict = None) -> Dict[str, str]:
    """
    Ingest nodes and relationships into Neo4j.
    """
    neo4j_client = driver or neo4j_driver
    
    with neo4j_client.session() as session:
        # Crea nodi
        for name, node_id in nodes.items():
            # Determina il tipo di nodo dal nome
            if name.startswith("BANDO:"):
                label = "Bando"
                # Aggiungi attributi specifici del bando
                additional_props = {}
                if bando_attrs:
                    # Filtra gli attributi None
                    for k, v in bando_attrs.items():
                        if v is not None:
                            # Converti i tipi per Neo4j
                            if isinstance(v, (list, dict)):
                                additional_props[k] = json.dumps(v, ensure_ascii=False)
                            else:
                                additional_props[k] = v
            elif name.startswith("DIMENSIONE:"):
                label = "Dimensione"
                additional_props = {}
            elif name.startswith("INTERVENTO"):
                label = "Intervento"
                additional_props = {}
            elif name.startswith("CRITERIO"):
                label = "Criterio"
                additional_props = {}
            elif name.startswith("REQUISITO:"):
                label = "Requisito"
                additional_props = {}
            elif name.startswith("LOCALIZZAZIONE:"):
                label = "Localizzazione"
                additional_props = {}
            elif name.startswith("MASSIMALE"):
                label = "Massimale"
                additional_props = {}
            elif name.startswith("IMPORTO"):
                label = "Importo"
                additional_props = {}
            elif name.startswith("ATTRIBUTO_BONUS:"):
                label = "AttributoBonus"
                additional_props = {}
            else:
                label = "Entity"
                additional_props = {}
            
            # Crea la query CREATE con tutti i parametri
            props_dict = {"id": node_id, "name": name}
            props_dict.update(additional_props)
            
            props_str = ", ".join([f"{k}: ${k}" for k in props_dict.keys()])
            
            session.run(
                f"""
                CREATE (n:{label} {{{props_str}}})
                """,
                **props_dict
            )
        
        # Crea relazioni
        for rel in relationships:
            session.run(
                """
                MATCH (a {id: $source_id}), (b {id: $target_id})
                CREATE (a)-[r:RELATIONSHIP {type: $type}]->(b)
                SET r += $attributes
                """,
                source_id=rel["source"],
                target_id=rel["target"],
                type=rel["type"],
                attributes=rel.get("attributes", {}),
            )
    
    print(f"[NEO4J] Ingestiti {len(nodes)} nodi e {len(relationships)} relazioni")
    return nodes

# =========================
# INGESTIONE IN QDRANT CON COLLEGAMENTI AL GRAFO
# =========================

def create_collection(client, collection_name: str, vector_dimension: int):
    """Crea la collection Qdrant solo se non esiste."""
    try:
        client.get_collection(collection_name)
        print(f"[QDRANT] Collection '{collection_name}' già esistente")
    except Exception as e:
        if "Not found" in str(e):
            print(f"[QDRANT] Creazione collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE
                )
            )
            print(f"[QDRANT] Collection '{collection_name}' creata")
        else:
            raise

def ingest_to_qdrant(
    collection_name: str,
    chunks: List[str],
    ontology: BandoOntology,
    nodes_dict: Dict[str, str]
):
    """
    Ingestione in Qdrant:
    - chunks: lista di stringhe (testo chunk)
    - ontology: ontologia del bando
    - nodes_dict: mappatura nome_nodo -> id_univoco (da Neo4j)
    
    Ogni punto Qdrant contiene:
    - text: il testo del chunk
    - nodes: lista di id Neo4j a cui è collegato (inizialmente solo il bando)
    - bando_id: l'id del bando principale
    """
    # Calcola embedding per tutti i chunk
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    points = []
    bando_id = ontology.id
    bando_name = f"BANDO: {ontology.identificativo}"
    
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # Ogni chunk è collegato al bando principale
        node_ids = [bando_id]
        
        # Cerca keywords nel chunk per collegamenti aggiuntivi
        chunk_lower = chunk.lower()
        
        # Collegamento a dimensioni
        for dim in ontology.dimensioni_ammesse:
            if dim.lower() in chunk_lower and f"DIMENSIONE: {dim}" in nodes_dict:
                node_ids.append(nodes_dict[f"DIMENSIONE: {dim}"])
        
        # Collegamento a interventi
        for intervento in ontology.interventi:
            nome = intervento.get("nome", "").lower()
            tipo = intervento.get("tipo", "")
            if nome and nome in chunk_lower and f"INTERVENTO {tipo}: {intervento.get('nome')}" in nodes_dict:
                node_ids.append(nodes_dict[f"INTERVENTO {tipo}: {intervento.get('nome')}"])
        
        # Collegamento a criteri
        for criterio in ontology.criteri_selezione:
            nome = criterio.get("nome", "").lower()
            codice = criterio.get("codice", "")
            if nome and nome in chunk_lower and f"CRITERIO {codice}: {criterio.get('nome')}" in nodes_dict:
                node_ids.append(nodes_dict[f"CRITERIO {codice}: {criterio.get('nome')}"])
        
        # Rimuovi duplicati
        node_ids = list(set(node_ids))
        
        payload = {
            "text": chunk,
            "nodes": node_ids,
            "bando_id": bando_id,
            "bando_name": ontology.identificativo,
            "chunk_index": idx,
            "chunk_total": len(chunks)
        }
        
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload,
            )
        )
    
    # Inserisci in Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )   
    
    print(f"[QDRANT] Inseriti {len(points)} punti")
    return len(points)

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
    
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k * 5,  # Prendi più risultati per avere più nodi
    )
    
    # 2. Estrai nodi dai risultati Qdrant
    all_node_ids = set()
    qdrant_texts = []
    
    for point in search_results:
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


# =========================
# GRAPH CONTEXT & QA CON REGOLO
# =========================

def fetch_related_graph(neo4j_client, entity_ids: List[str]) -> List[Dict]:
    """
    Recupera il subgraph con tutti gli attributi dei nodi.
    """
    if not entity_ids:
        return []
    
    query = """
    MATCH (e)
    WHERE e.id IN $entity_ids
    OPTIONAL MATCH (e)-[r1]-(n1)
    WHERE type(r1) IS NOT NULL
    OPTIONAL MATCH (n1)-[r2]-(n2)
    WHERE type(r2) IS NOT NULL
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
    """
    
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        
        for record in result:
            # Verifica che il nodo sorgente esista
            if not record["source_node"]:
                continue
                
            # Aggiungi relazione 1 se esiste
            if record["rel1_type"] and record["target1_node"]:
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
            
            # Aggiungi relazione 2 se esiste
            if record["rel2_type"] and record["target2_node"]:
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
    
    print(f"[DEBUG] Subgraph recuperato con {len(subgraph)} relazioni")
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
        for i, text in enumerate(qdrant_texts[:3])  # Limita a 3 per brevità
    ]) if qdrant_texts else "Nessun riferimento testuale trovato."
    
    return {
        "nodes": nodes_formatted,
        "edges": edges_info,
        "qdrant_context": qdrant_context,
        "nodes_info": nodes_info,
        "edges_info": edges_info
    }

def graphRAG_run(graph_context, user_query: str) -> str:
    """Versione base senza history"""
    nodes_str = "\n".join(graph_context["nodes"])
    
    # Correzione anche qui
    edges_info = graph_context.get("edges", [])
    edges_list = []
    
    for edge in edges_info:
        edge_str = f"{edge.get('from', '?')} --[{edge.get('type', '?')}]--> {edge.get('to', '?')}"
        attrs = edge.get('attributes', {})
        if attrs:
            attrs_str = ", ".join([f"{k}: {v}" for k, v in attrs.items()])
            edge_str += f" ({attrs_str})"
        edges_list.append(edge_str)
    
    edges_str = "; ".join(edges_list)
    
    qdrant_context = graph_context.get("qdrant_context", "")
    
    prompt = f"""
Sei un assistente che risponde usando SOLO il seguente grafo di conoscenza come contesto.

NODI (ogni nodo ha i suoi attributi elencati sotto):
{nodes_str}

ARCHI (relazioni tra nodi):
{edges_str}

RIFERIMENTI DAL DOCUMENTO:
{qdrant_context}

Domanda dell'utente:
"{user_query}"

Analizza attentamente gli attributi dei nodi e le relazioni tra di essi.
Rispondi in modo sintetico, preciso e aderente al grafo, CITANDO ESPLICITAMENTE GLI ATTRIBUTI DEI NODI quando rilevanti per la risposta.
"""
    task = regolo_call.delay(prompt)
    response = task.get(timeout=400)
    
    return response


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
    print("Contesto Qdrant:", qdrant_context[:500] + "..." if len(qdrant_context) > 500 else qdrant_context)
    
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
    task = regolo_call.delay(prompt)
    response = task.get(timeout=400)
    
    return response

from dotenv import load_dotenv
load_dotenv()

from celery import Celery
import os
import requests
import json
import fitz
import re
import datetime
import uuid
import hashlib
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient, models
from celery import group, chain

from doctr.io import read_pdf
from doctr.models import ocr_predictor

# ==========================
# OpenAI / Regolo config
# ==========================

REGOLO_API_URL = "https://api.openai.com/v1/chat/completions"
REGOLO_KEY = os.getenv("OPENAI_API_KEY")

# ⚠️ USA UN MODELLO VALIDO
REGOLO_MODEL = "gpt-5-nano"
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

if not REGOLO_KEY:
    raise RuntimeError("OPENAI_API_KEY non impostata")

class GraphEntry(BaseModel):
    node: str
    target_node: str | None
    relationship: str | None
    attributes: dict[str, Any] | None = None


class GraphComponents(BaseModel):
    graph: list[GraphEntry]


# Ontologia 

class BandoOntology(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # A. Avviso
    identificativo: str
    autorita: str  
    soggetto_attuatore: Optional[str] = None   
    dotazione_finanziaria: Optional[float] = None  
    regime_aiuto: Optional[str] = None 
    
    # NUOVE CATEGORIE - Liste per possibili multiple
    tematiche: List[str] = []  # Può essere più contemporaneamente
    tipologie_fondo: List[str] = []  # Europei, Nazionali, Privati, Regionali
    tipologie_agevolazione: List[str] = []  # A fondo perduto, Credito d'imposta, ecc.
    destinatari: List[str] = []  # Aggregazioni d'impresa, Donne, Giovani, ecc.
    
    # Finestre temporali
    data_apertura_formulario: Optional[str] = None 
    data_inizio_domande: Optional[str] = None 
    data_fine_domande: Optional[str] = None  
    
    piattaforma_presentazione: Optional[str] = None 
    
    # B. Beneficiario
    dimensioni_ammesse: List[str] = []  
    requisiti_beneficiario: List[str] = []  
    sede_obbligatoria: Optional[str] = None  # Diventa nodo unico Localizzazione
    
    # Attributi bonus
    attributi_bonus: List[str] = []  
    
    # C. Progetto
    tipologie_intervento: List[str] = []  
    sottotipologie_cloud: List[str] = []  
    importo_minimo_progetto: Optional[float] = None  
    tempo_realizzazione_mesi: Optional[int] = None  
    
    # Esclusioni specifiche
    esclusioni: List[str] = []  
    
    # D. Intervento (dettagli per tipologia)
    interventi: List[dict] = []  
    
    # E. Contributo
    natura_contributo: Optional[str] = None  
    importi_unitari: dict = {}  
    massimali_dimensione: dict = {}  
    incompatibilita_aiuti: bool = False
    
    # F. Criteri di Selezione
    criteri_selezione: List[dict] = []  
    criterio_tiebreak: Optional[str] = None
    
    # G. Procedura Domanda
    allegati_obbligatori: List[str] = []
    piattaforma_invio: Optional[str] = None
    regola_senza_soccorso: bool = False
    
    # H. Istruttoria
    ordine_istruttoria: Optional[str] = None  
    
    # I. Erogazione
    modalita_erogazione: Optional[str] = None
    documenti_erogazione: List[str] = []
    
    # L. Revoca/Rinuncia
    condizioni_revoca: List[str] = []
    
    # Metadata
    file_name: Optional[str] = None
    pagine_totali: Optional[int] = None
    data_processing: str = Field(default_factory=lambda: datetime.now().isoformat())
    fonte: Optional[str] = None 

# ==========================
# Celery config
# ==========================

doctr_ocr = ocr_predictor(
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_vgg16_bn",
    pretrained=True,
)

celery_app = Celery(
    "celery_app",
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

# ==========================
# HTTP headers
# ==========================

HEADERS = {
    "Authorization": f"Bearer {REGOLO_KEY}",
    "Content-Type": "application/json",
}

# ==========================
# Funzioni di supporto (NON decorate)
# ==========================

def regolo_call(prompt: str, response_format: str = "text") -> str:
    """
    Chiama OpenAI Chat Completions
    response_format: "text" | "json"
    """

    payload = {
        "model": REGOLO_MODEL,
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

def is_scanned_page(page: fitz.Page, text_min_chars: int = 80) -> bool:
    """Ritorna True se la pagina sembra scansionata."""
    text = page.get_text("text") or ""                                        
    has_images = len(page.get_images(full=True)) > 0
    return (len(text.strip()) < text_min_chars) and has_images

def doctr_ocr_per_page(pdf_path: str) -> list[str]:               
    """Usa DocTR per fare OCR dell'intero PDF."""
    doc = read_pdf(pdf_path)                                      #Flusso per scansionamento se serve
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
    #Estrae testo da PDF con fallback OCR per pagine scansionate
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
    """Chunking basato su paragrafi con fallback word-based."""
    # Separa il testo in paragrafi
    paragraphs = text.split('\n')
    chunks: list[str] = []
    current_chunk_words: list[str] = []
    current_word_count = 0
    
    for para in paragraphs:
        if not para.strip():  # Salta paragrafi vuoti
            continue
            
        para_words = para.split()
        para_word_count = len(para_words)
        
        # Se il paragrafo da solo supera max_words, usiamo chunking word-based su di esso
        if para_word_count > max_words:
            # Prima aggiungiamo ciò che abbiamo accumulato finora
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
                current_word_count = 0
            
            # Chunking word-based sul paragrafo lungo
            words = para.split()
            n = len(words)
            start = 0
            while start < n:
                end = min(start + max_words, n)
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                if end == n:
                    break
                start = end - overlap_words
            continue
        
        # Se aggiungendo questo paragrafo superiamo il limite
        if current_word_count + para_word_count > max_words:
            if current_chunk_words:  # Aggiungi il chunk corrente
                chunks.append(" ".join(current_chunk_words))
                
                # Prepara il prossimo chunk con overlap
                if overlap_words > 0:
                    # Prendi le ultime overlap_words parole dal chunk corrente
                    overlap_start = max(0, len(current_chunk_words) - overlap_words)
                    current_chunk_words = current_chunk_words[overlap_start:]
                    current_word_count = len(current_chunk_words)
                else:
                    current_chunk_words = []
                    current_word_count = 0
        
        # Aggiungi il paragrafo al chunk corrente
        if current_chunk_words:
            current_chunk_words.append(para)
        else:
            current_chunk_words = [para]
        current_word_count += para_word_count
    
    # Aggiungi l'ultimo chunk se non è vuoto
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    
    # Se non siamo riusciti a fare chunking per paragrafi (testo senza newline),
    # usiamo il metodo word-based originale
    if not chunks and text.strip():
        words = text.split()
        n = len(words)
        start = 0
        while start < n:
            end = min(start + max_words, n)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == n:
                break
            start = end - overlap_words
    
    return chunks

def build_ontology_prompt_complete(full_text: str) -> str:
    """Crea il prompt per estrarre l'ontologia completa con nuove categorie."""
    return f"""
Sei un esperto analista di bandi di finanziamento. 
Analizza il seguente bando e estrai TUTTE le informazioni strutturate seguendo lo schema JSON sottostante.

**ISTRUZIONI CRITICHE:**
1. Estratti SOLO informazioni presenti nel testo. Non inventare nulla.
2. Per le date, converti in formato ISO 8601 se possibile: "16/10/2025 h 12:00" → "2025-10-16T12:00:00"
3. Per gli importi, rimuovi '€' e converti in numero float: "€ 15.000.000" → 15000000.0
4. Per le liste, includi tutti gli elementi rilevanti trovati nel testo.
5. Se un campo non è presente, lascia null o lista vuota.
6. Identificativo è il nome del bando

**NUOVE CATEGORIE IMPORTANTI:**

6. **TEMATICHE** (può essere più di una contemporaneamente):
   - Scegli tra queste opzioni standardizzate:
     "Accesso al credito", "Aerospazio", "Agenda 2030", "Agrifood", 
     "Animazione Territoriale", "Artigianato", "Cinema", "Cooperazione", 
     "Cooperazione internazionale", "Creatività", "Creazione di impresa", 
     "Cultura", "Digitalizzazione", "Economia Circolare", "Economia del mare", 
     "Editoria", "Formazione", "Gaming", "Green Economy", "ICT", "Inclusione", 
     "Innovazione", "Internazionalizzazione", "Investimenti", "Logistica", 
     "Mobilità", "Moda", "Next Generation Lazio", "Pari Opportunità", 
     "Pre-seed", "Progettazione europea", "Prototipazione", 
     "Ricerca/R&S", "S3", "Scale-up", "Scienza della vita", "Scuola", 
     "Sicurezza", "Sisma", "Smart City", "Sostegno a imprese in crisi", 
     "Sostenibilità", "Spettacolo dal vivo", "Sport", "Sviluppo Locale", 
     "Teatri", "Transizione ecologica", "Turismo", "Violenza di genere"
   - Seleziona TUTTE quelle rilevanti (anche più di una)
   - NON INSERIRE TEMATICHE AL DI FUORI DI QUESTE

7. **TIPOLOGIA FONDO** (può essere più di una):
   - "Europei", "Nazionali", "Privati", "Regional"

8. **TIPOLOGIA AGEVOLAZIONE** (può essere più di una):
   - "A Fondo perduto", "Credito d'imposta", "Equity Crowdfunding", 
     "Finanziamenti agevolati", "Finanziamenti tasso zero", "Garanzie", 
     "Microfinanza", "Premi in denaro e servizi", "Servizi", "Venture Capital"

9. **DESTINATARI** (può essere più di una):
   - "Aggregazioni d'impresa", "Altri enti", "CER", "Donne", "Giovani", 
     "Grandi imprese", "Imprese", "Imprese costituende", "Liberi professionisti", 
     "Maker", "Medie Imprese", "Micro-imprese", "No-profit", "Persone fisiche", 
     "Persone giuridiche", "Piccole imprese", "Pubblica Amministrazione", 
     "Ricercatori", "Scuole e studenti", "Startup e spinoff", 
     "Startup innovative", "Università e ricerca"

10. **SEDE OBBLIGATORIA** (Regione, anche se non specificata):
    - NON PUò essere null
    - Solo il nome della regione: ES: "Lazio", non "Regione Lazio". MA POSSONO ESSERCI ALTRE REGIONI
    - Per stati: "Italia", non "Stato italiano"

**STRUTTURA JSON OBBLIGATORIA:**
{{
  "identificativo": "string", 
  "autorita": "string", 
  "tematiche": ["lista di stringhe"],  # NUOVO
  "tipologie_fondo": ["lista di stringhe"],  # NUOVO
  "tipologie_agevolazione": ["lista di stringhe"],  # NUOVO
  "destinatari": ["lista di stringhe"],  # NUOVO
  "soggetto_attuatore": "string | null",
  "dotazione_finanziaria": "number | null",
  "regime_aiuto": "string | null",
  "data_apertura_formulario": "string | null",
  "data_inizio_domande": "string | null",
  "data_fine_domande": "string | null",
  "piattaforma_presentazione": "string | null",
  "dimensioni_ammesse": ["lista di stringhe"],
  "requisiti_beneficiario": ["lista di stringhe"],
  "sede_obbligatoria": "string",
  "attributi_bonus": ["lista di stringhe"],
  "tipologie_intervento": ["lista di stringhe"],
  "importo_minimo_progetto": "number | null",
  "tempo_realizzazione_mesi": "number | null",
  "esclusioni": ["lista di stringhe"],
  "interventi": [
    {{
      "tipo": "string",
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
    response = regolo_call(prompt)
    print(response)
    
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
        print(f"[ERROR] JSON ricevuto: {json_str[:1500]}")
        raise

def create_ontology_graph(ontology: BandoOntology) -> Tuple[Dict[str, str], List[Dict], Dict]:
    """
    Crea un grafo strutturato dall'ontologia del bando.
    """
    nodes: Dict[str, str] = {}
    relationships: List[Dict] = []
    
    # 1. NODO PRINCIPALE DEL BANDO
    bando_id = ontology.id
    bando_name = f"BANDO: {ontology.identificativo}"
    nodes[bando_name] = bando_id
    
    bando_attrs = {
        "identificativo": ontology.identificativo,
        "autorita": ontology.autorita,
        "dotazione_finanziaria": ontology.dotazione_finanziaria,
        "regime_aiuto": ontology.regime_aiuto,
        "data_fine_domande": ontology.data_fine_domande,
        "piattaforma_presentazione": ontology.piattaforma_presentazione,
        "importo_minimo_progetto": ontology.importo_minimo_progetto,
        "tempo_realizzazione_mesi": ontology.tempo_realizzazione_mesi,
        "file_name": ontology.file_name,
       
    }
    bando_attrs = {k: v for k, v in bando_attrs.items() if v is not None}

    def create_unique_node(label: str, value: str, prefix: str = None) -> str:
        """Crea o restituisce ID per un nodo unico basato su hash."""
        if prefix:
            node_name = f"{prefix}: {value}"
        else:
            node_name = f"{label.upper()}: {value}"
        
        node_hash = hashlib.md5(f"{label}:{value}".encode()).hexdigest()[:8]
        node_id = f"{label.upper()[:4]}_{node_hash}"
        
        if node_name not in nodes:
            nodes[node_name] = node_id
        
        return node_id 
    # 3. NODI TEMATICHE (multiple, ognuna unica)
    for tematica in ontology.tematiche:
        tematica_id = create_unique_node("Tematica", tematica)
        relationships.append({
            "source": bando_id,
            "target": tematica_id,
            "type": "RICONOSCE_TEMATICA",
            "attributes": {}
        })
    
    # 4. NODI TIPOLOGIA FONDO (multiple, ognuna unica)
    for fondo in ontology.tipologie_fondo:
        fondo_id = create_unique_node("TipologiaFondo", fondo)
        relationships.append({
            "source": bando_id,
            "target": fondo_id,
            "type": "FINANZIATO_DA",
            "attributes": {}
        })
    
    # 5. NODI TIPOLOGIA AGEVOLAZIONE (multiple, ognuna unica)
    for agevolazione in ontology.tipologie_agevolazione:
        agevolazione_id = create_unique_node("TipologiaAgevolazione", agevolazione)
        relationships.append({
            "source": bando_id,
            "target": agevolazione_id,
            "type": "OFFRE_AGEVOLAZIONE",
            "attributes": {}
        })
    
    # 6. NODI DESTINATARI (multiple, ognuna unica)
    for destinatario in ontology.destinatari:
        destinatario_id = create_unique_node("Destinatario", destinatario)
        relationships.append({
            "source": bando_id,
            "target": destinatario_id,
            "type": "RIVOLTO_A",
            "attributes": {}
        })
    
    # 7. NODO LOCALIZZAZIONE (UNICO)
    if ontology.sede_obbligatoria:
        loc_id = create_unique_node("Localizzazione", ontology.sede_obbligatoria)
        relationships.append({
            "source": bando_id,
            "target": loc_id,
            "type": "LIMITATO_A_AREA",
            "attributes": {"area": ontology.sede_obbligatoria}
        })
    
    # 4. NODI PER DIMENSIONI AMMESSE (questi possono rimanere duplicati se necessario)
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
    
    # 5. NODI PER TIPOLOGIE DI INTERVENTO
    for intervento in ontology.interventi:
        tipo = intervento.get("tipo", "")
        nome = intervento.get("nome", "")
        
        if tipo and nome:
            intervento_name = f"INTERVENTO {tipo}: {nome}"
            intervento_id = str(uuid.uuid4())
            nodes[intervento_name] = intervento_id
            
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
            
            # COLGAMENTO IMPORTI
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
    
    # 6. NODI PER CRITERI DI SELEZIONE
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
    
    # 7. NODI PER REQUISITI
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
    
    # 8. NODI PER MASSIMALI
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
    
    # 9. NODI PER ATTRIBUTI BONUS
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

def ingest_to_neo4j(nodes: Dict[str, str], relationships: List[Dict], driver=None, bando_attrs: Dict = None) -> Dict[str, str]:
    """
    Ingest nodes and relationships into Neo4j.
    Tutti i nodi delle nuove categorie usano MERGE per essere unici.
    """
    neo4j_client = driver or neo4j_driver
    
    # Definizione dei tipi di nodo che devono essere unici (MERGE)
    UNIQUE_NODE_PREFIXES = {
        "TEMATICA:": "Tematica", 
        "TIPOLOGIAFONDO:": "TipologiaFondo",
        "TIPOLOGIAAGEVOLAZIONE:": "TipologiaAgevolazione",
        "DESTINATARIO:": "Destinatario",
        "LOCALIZZAZIONE:": "Localizzazione"
    }
    
    with neo4j_client.session() as session:
        # Crea nodi
        for name, node_id in nodes.items():
            # Determina il tipo di nodo
            label = None
            is_unique = False
            
            for prefix, node_label in UNIQUE_NODE_PREFIXES.items():
                if name.startswith(prefix):
                    label = node_label
                    is_unique = True
                    break
            
            if not label:
                # Nodo non unico, determina label dal contenuto
                if name.startswith("BANDO:"):
                    label = "Bando"
                elif name.startswith("DIMENSIONE:"):
                    label = "Dimensione"
                elif name.startswith("INTERVENTO"):
                    label = "Intervento"
                elif name.startswith("CRITERIO"):
                    label = "Criterio"
                elif name.startswith("REQUISITO:"):
                    label = "Requisito"
                elif name.startswith("MASSIMALE"):
                    label = "Massimale"
                elif name.startswith("IMPORTO"):
                    label = "Importo"
                elif name.startswith("ATTRIBUTO_BONUS:"):
                    label = "AttributoBonus"
                else:
                    label = "Entity"
            
            # Proprietà del nodo
            props_dict = {"id": node_id, "name": name}
            
            # Aggiungi attributi specifici per il bando
            if label == "Bando" and bando_attrs:
                for k, v in bando_attrs.items():
                    if v is not None:
                        if isinstance(v, (list, dict)):
                            props_dict[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            props_dict[k] = v
            
            props_str = ", ".join([f"{k}: ${k}" for k in props_dict.keys()])
            
            # Usa MERGE per nodi unici, CREATE per nodi specifici
            if is_unique:
                query = f"""
                MERGE (n:{label} {{id: $id}})
                ON CREATE SET n = {{{props_str}}}
                ON MATCH SET n.name = $name
                """
            else:
                query = f"CREATE (n:{label} {{{props_str}}})"
            
            session.run(query, **props_dict)
        
        # Crea relazioni (uguale al precedente)
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
    Ingestione in Qdrant con collegamenti a tutti i nodi unici del grafo Neo4j.
    """
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    points = []
    bando_id = ontology.id
    bando_name = f"BANDO: {ontology.identificativo}"
    
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        node_ids = [bando_id]
        chunk_lower = chunk.lower()
        
        # Funzione helper per aggiungere nodi unici
        def add_unique_node_if_relevant(node_type: str, values: List[str], keywords: List[str] = None):
            for value in values:
                node_name = f"{node_type.upper()}: {value}"
                if node_name in nodes_dict:
                    # Se ci sono keywords specifiche, controllale
                    if keywords:
                        for keyword in keywords:
                            if keyword in chunk_lower:
                                node_ids.append(nodes_dict[node_name])
                                break
                    else:
                        # Altrimenti aggiungi sempre se il valore è nel chunk
                        if value.lower() in chunk_lower:
                            node_ids.append(nodes_dict[node_name])
        
        # 1. TEMATICHE
        tematiche_keywords = ["tematic", "settore", "ambito", "focus", "area", "settorial", "priorit"]
        add_unique_node_if_relevant("TEMATICA", ontology.tematiche, tematiche_keywords)
        
        # 2. TIPOLOGIE FONDO
        fondo_keywords = ["fondo", "finanziamento", "programma", "fonte", "risorse", "finanziar", "budget"]
        add_unique_node_if_relevant("TIPOLOGIAFONDO", ontology.tipologie_fondo, fondo_keywords)
        
        # 3. TIPOLOGIE AGEVOLAZIONE
        agevolazione_keywords = [
            "agevolazione", "contributo", "incentivo", "finanziamento", "fondo perduto", 
            "credito d'imposta", "garanzia", "premio", "servizi", "finanziamenti agevolati",
            "tasso zero", "microfinanza", "equity", "venture capital"
        ]
        add_unique_node_if_relevant("TIPOLOGIAAGEVOLAZIONE", ontology.tipologie_agevolazione, agevolazione_keywords)
        
        # 4. DESTINATARI
        destinatario_keywords = [
            "destinatari", "beneficiari", "rivolto a", "possono partecipare", "ammissibili",
            "soggetti", "richiedenti", "partecipanti", "imprese", "startup", "pmi",
            "professionisti", "enti", "associazioni", "cooperative"
        ]
        add_unique_node_if_relevant("DESTINATARIO", ontology.destinatari, destinatario_keywords)
        
        # 5. LOCALIZZAZIONE
        if ontology.sede_obbligatoria:
            loc_name = f"LOCALIZZAZIONE: {ontology.sede_obbligatoria}"
            if loc_name in nodes_dict:
                # Cerca regione nel chunk (case insensitive)
                if any(word.lower() in chunk_lower for word in ontology.sede_obbligatoria.split()):
                    node_ids.append(nodes_dict[loc_name])
        
        # 6. DIMENSIONI AMMESSE
        for dimensione in ontology.dimensioni_ammesse:
            dim_name = f"DIMENSIONE: {dimensione}"
            if dim_name in nodes_dict and dimensione.lower() in chunk_lower:
                node_ids.append(nodes_dict[dim_name])
        
        # 7. INTERVENTI
        for intervento in ontology.interventi:
            tipo = intervento.get("tipo", "")
            nome = intervento.get("nome", "")
            if tipo and nome:
                intervento_name = f"INTERVENTO {tipo}: {nome}"
                if intervento_name in nodes_dict:
                    # Cerca termini relativi all'intervento nel chunk
                    if any(term.lower() in chunk_lower for term in [tipo, nome, "intervento", "azione", "progetto"]):
                        node_ids.append(nodes_dict[intervento_name])
        
        # 8. CRITERI DI SELEZIONE
        for criterio in ontology.criteri_selezione:
            codice = criterio.get("codice", "")
            nome = criterio.get("nome", "")
            if codice and nome:
                criterio_name = f"CRITERIO {codice}: {nome}"
                if criterio_name in nodes_dict:
                    if any(term.lower() in chunk_lower for term in [codice, nome, "criterio", "punteggio", "valutazione"]):
                        node_ids.append(nodes_dict[criterio_name])
        
        # 9. REQUISITI BENEFICIARIO
        for requisito in ontology.requisiti_beneficiario:
            req_name = f"REQUISITO: {requisito}"
            if req_name in nodes_dict:
                if requisito.lower() in chunk_lower or any(word in chunk_lower for word in ["requisito", "condizione", "obbligo"]):
                    node_ids.append(nodes_dict[req_name])
        
        # 10. MASSIMALI DIMENSIONE
        if isinstance(ontology.massimali_dimensione, dict):
            for dimensione, importo in ontology.massimali_dimensione.items():
                if importo:
                    massimale_name = f"MASSIMALE {dimensione}: €{importo}"
                    if massimale_name in nodes_dict:
                        if any(term.lower() in chunk_lower for term in [dimensione, "massimale", "limite", "importo massimo"]):
                            node_ids.append(nodes_dict[massimale_name])
        
        # 11. ATTRIBUTI BONUS
        for attributo in ontology.attributi_bonus:
            attributo_name = f"ATTRIBUTO_BONUS: {attributo}"
            if attributo_name in nodes_dict:
                if attributo.lower() in chunk_lower or "bonus" in chunk_lower:
                    node_ids.append(nodes_dict[attributo_name])
        
        # 12. IMPORTI UNITARI (collegati agli interventi)
        for intervento in ontology.interventi:
            tipo = intervento.get("tipo", "")
            if tipo and tipo in ontology.importi_unitari:
                importi = ontology.importi_unitari[tipo]
                if isinstance(importi, dict):
                    for dimensione, importo in importi.items():
                        if importo:
                            importo_name = f"IMPORTO {tipo} {dimensione}: €{importo}"
                            if importo_name in nodes_dict and (str(importo) in chunk or "importo" in chunk_lower):
                                node_ids.append(nodes_dict[importo_name])
                elif isinstance(importi, (int, float)) and importi > 0:
                    importo_name = f"IMPORTO {tipo}: €{importi}"
                    if importo_name in nodes_dict and (str(importi) in chunk or "importo" in chunk_lower):
                        node_ids.append(nodes_dict[importo_name])
        
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
        
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload=payload,
        ))
    
    # Invia i punti a Qdrant
    create_collection(qdrant_client, collection_name, VECTOR_DIM)
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"[QDRANT] Ingestiti {len(points)} vettori nella collezione '{collection_name}'")
    print(f"[QDRANT] Collegati a {len(set().union(*[p.payload.get('nodes', []) for p in points]))} nodi Neo4j unici")

# ==========================
# Task principale decorato
# ==========================

processing_status = {}

@celery_app.task(bind=True, max_retries=3)
def process_document_pipeline(self, job_id: str, pdf_path: str, collection_name: str = "Bandi"):
    """
    Task principale decorato che esegue tutta la pipeline in serie.
    Il parallelismo sarà gestito a livello superiore chiamando questo task in gruppo.
    """
    try:
        # Inizializza stato
        self.update_state(
            state='PROGRESS', 
            meta={
                'status': 'processing', 
                'progress': 'Lettura PDF...',
                'job_id': job_id
            }
        )
        
        # 1. Estrazione testo
        raw_data = extract_pdf_text_with_tables(pdf_path)
        
        # 2. Normalizzazione
        clean_data = normalize_whitespace(raw_data)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Estrazione ontologia strutturata (1 chiamata LLM)...',
                'job_id': job_id
            }
        )
        
        # 3. Estrazione ontologia
        ontology = extract_ontology_from_text(clean_data)
        ontology.file_name = os.path.basename(pdf_path)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Creazione grafo ontologico...',
                'job_id': job_id,
                'details': {
                    'identificativo': ontology.identificativo,
                    'dimensioni_ammesse': ontology.dimensioni_ammesse,
                    'tipologie_intervento': ontology.tipologie_intervento,
                    'criteri_count': len(ontology.criteri_selezione)
                }
            }
        )
        
        # 4. Creazione grafo
        nodes, relationships, bando_attrs = create_ontology_graph(ontology)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Creazione chunk per ricerca semantica...',
                'job_id': job_id,
                'graph_created': True
            }
        )
        
        # 5. Chunking
        chunks = chunk_text(clean_data, max_words=250, overlap_words=35)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Salvataggio in database...',
                'job_id': job_id,
                'chunks_created': len(chunks)
            }
        )
        
        # 6. ESECUZIONE IN SERIE: Neo4j + Qdrant
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Salvataggio in Neo4j...',
                'job_id': job_id
            }
        )
        
        # Crea collection Qdrant se non esiste
        create_collection(qdrant_client, collection_name, VECTOR_DIM)
        
        # Salva in Neo4j
        ingest_to_neo4j(
            nodes=nodes,
            relationships=relationships,
            bando_attrs=bando_attrs
        )
        
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Salvataggio in Qdrant...',
                'job_id': job_id
            }
        )
        
        # Salva in Qdrant
        ingest_to_qdrant(
            collection_name=collection_name,
            chunks=chunks,
            ontology=ontology,
            nodes_dict=nodes
        )
        
        # 7. Salva metadata e ontologia
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'processing',
                'progress': 'Salvataggio metadata...',
                'job_id': job_id
            }
        )
        
        # Salva l'ontologia per riferimento
        STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        ontology_file = os.path.join(STORAGE_DIR, f"{job_id}_ontology.json")
        with open(ontology_file, 'w') as f:
            json.dump(ontology.model_dump(), f, indent=2, default=str)
        
        # Risultato finale
        result = {
            "status": "completed",
            "progress": "Completato!",
            "job_id": job_id,
            "collection_name": collection_name,
            "chunks_count": len(chunks),
            "nodes_count": len(nodes),
            "relationships_count": len(relationships),
            "qdrant_points": len(chunks),  # Numero di punti Qdrant = numero di chunks
            "ontology": {
                "identificativo": ontology.identificativo,
                "autorita": ontology.autorita,
                "dotazione": ontology.dotazione_finanziaria,
                "interventi": len(ontology.interventi),
                "criteri": len(ontology.criteri_selezione)
            },
            "bando_attributes": bando_attrs
        }
        
        # Aggiorna stato globale
        if 'processing_status' in globals():
            processing_status[job_id] = result
        
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "progress": "Errore!",
            "job_id": job_id,
            "error": str(e)
        }
        
        if 'processing_status' in globals():
            processing_status[job_id] = error_result
        
        raise self.retry(exc=e, countdown=60)
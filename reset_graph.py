import os
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================

load_dotenv()

qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# NOME DELLA COLLECTION QDRANT DA SVUOTARE
COLLECTION = "Bandi"

# =========================
# QDRANT DELETE
# =========================

def reset_qdrant():
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key
    )
    print(f"[QDRANT] Eliminazione collection '{COLLECTION}'...")
    try:
        client.delete_collection(COLLECTION)
        print("[QDRANT] Collection rimossa con successo.")
    except Exception as e:
        print(f"[QDRANT] Errore: {e}")


# =========================
# NEO4J DELETE
# =========================

def reset_neo4j():
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    print("[NEO4J] Cancellazione nodi + relazioni...")

    query = """
    MATCH (n)
    DETACH DELETE n
    """

    try:
        with driver.session() as session:
            session.run(query)
        print("[NEO4J] Tutto cancellato correttamente.")
    except Exception as e:
        print(f"[NEO4J] Errore: {e}")
    finally:
        driver.close()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("=== RESET GRAPHRAG ===")
    reset_qdrant()
    reset_neo4j()
    print("=== COMPLETATO ===")

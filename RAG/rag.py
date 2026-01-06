from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Lire de fichier texte
text = Path("equipements_surf.txt").read_text(encoding="utf-8")



### Chunking du texte
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

chunks = chunk_text(text)
print(f"{len(chunks)} chunks générés")


## Embeddings
model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(
    chunks,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


## Création de la base vectorielle Chroma
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",    # dossier où Chroma stocke les vecteurs
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(
    name="surf_rag"
)


# insesrtion des chunks + embeddings
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[
        {"source": "equipements_surf.txt"}
        for _ in chunks
    ]
)



## TEST de requête dans Chroma
query = "Quel équipement pour surfer en eau froide ?"

query_embedding = model.encode(
    query,
    convert_to_numpy=True,
    normalize_embeddings=True
)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

for doc in results["documents"][0]:
    print("\n---")
    print(doc[:300])

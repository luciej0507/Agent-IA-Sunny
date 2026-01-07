from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
from groq import Groq


# Lire de fichier texte
text = Path("equipements_surf.txt").read_text(encoding="utf-8")

# Modèle d'embedding
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# ChromaDB
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="surf_rag")

# Client Groq pour le RAG
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm_client = Groq(api_key=groq_api_key)

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


# Remplir la base ChromaDB avec les documents la première fois
def initialize_rag(text):
    """Initialise la base RAG avec le texte fourni"""
    # Vérifie si déjà initialisé
    if collection.count() > 0:
        print(f"Base déjà initialisée avec {collection.count()} chunks")
        return
    
    print("Initialisation de la base RAG...")
    
    # Chunking
    chunks = chunk_text(text)
    print(f"{len(chunks)} chunks générés")
    
    # Embeddings
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Stockage
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"{len(chunks)} chunks stockés dans ChromaDB")


# Fonction pour interroger le RAG
def ask_rag(query, n_results=3):
    """
    Fonction complète pour interroger ton RAG
    """
    # 1. Embedding de la question
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # 2. Recherche dans ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    # 3. Prépare le contexte
    context = "\n\n".join(results["documents"][0])
    
    # 4. Prompt
    prompt = f"""Tu es un assistant spécialisé en équipements de surf
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni

Contexte :
{context}

Question : {query}

Réponse :"""
    
    # 5. Appel LLM
    response = llm_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en surf."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content




# test avec boucle interactive
if __name__ == "__main__":
    print("Tapez 'quitter', 'exit' ou 'q' pour arrêter.\n")

    while True:
        # 1. Récupère la question de l'utilisateur
        user_query = input("Toi : ")

        # 2. Condition de sortie
        if user_query.lower() in ['quitter', 'exit', 'q', 'quit']:
            print("Bonne session de surf ! À bientôt.")
            break

        # 3. Vérifie que la question n'est pas vide
        if not user_query.strip():
            continue

        try:
            # 4. Appel de votre fonction ask_rag
            answer = ask_rag(query=user_query, n_results=3)

            # 5. Affichage du résultat
            print(f"RÉPONSE :\n{answer}")

        except Exception as e:
            print(f"Une erreur est survenue : {e}\n")
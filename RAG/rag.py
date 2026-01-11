import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb



# Modèle d'embedding
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ChromaDB
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="surf_rag")

# Client Groq pour le RAG
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm_client = Groq(api_key=groq_api_key)



def load_pdf(file_path):
    """Extrait le texte d'un PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def chunk_text(text, chunk_size=800, chunk_overlap=100):
    """
    Version améliorée qui tente de respecter la ponctuation.
    chunk_size ici est en caractères (plus précis pour les modèles d'embedding).
    """
    # 1. On nettoie un peu les sauts de ligne excessifs
    text = text.replace("\r", "").replace("\n\n", " [SPLIT] ").replace("\n", " ")
    
    # 2. On sépare par notre marqueur de paragraphe
    paragraphs = text.split(" [SPLIT] ")
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Si le paragraphe seul est plus petit que la limite, on l'ajoute
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + " "
        else:
            # Sinon, on stocke le chunk actuel et on recommence
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Gestion de l'overlap : on garde la fin du précédent pour le début du suivant
            current_chunk = para[-chunk_overlap:] if len(para) > chunk_overlap else ""
            current_chunk += para + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


## Remplir la base ChromaDB avec les documents la première fois
def initialize_rag():
    """Initialise la base RAG avec le texte fourni"""
    # Vérifie si déjà initialisé
    if collection.count() > 0:
        # print(f"Base déjà initialisée avec {collection.count()} chunks")
        return
    
    # print("Initialisation de la base RAG...")
    
    all_texts = []
    
    # 1. Chargement du fichier .txt (Equipements)
    if os.path.exists("equipements_surf.txt"):
        all_texts.append(Path("equipements_surf.txt").read_text(encoding="utf-8"))
        # print("TXT chargé")
        
    # 2. Chargement du PDF (Spots Bretagne - Anglais)
    if os.path.exists("Brittany-Surf-Guide.pdf"):
        all_texts.append(load_pdf("Brittany-Surf-Guide.pdf"))
        # print("PDF chargé")
    
    # Fusion et Chunking
    full_content = "\n\n".join(all_texts)
    chunks = chunk_text(full_content)
      
    # Embeddings
    embeddings = embedding_model.encode(
        chunks,
        # show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Stockage
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    # print(f"{len(chunks)} chunks stockés dans ChromaDB")


# Fonction pour interroger le RAG
def ask_rag(query, n_results=3):
    """
    Fonction complète pour interroger le RAG.
    Recherche les passages les plus pertinents dans la base de connaissances 
    (PDF spots Bretagne et TXT équipements).
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
    
    # 3. On extrait les textes trouvés
    documents = results.get("documents", [[]])[0]
    
    # 4. On les fusionne pour l'Agent
    # On peut ajouter une petite balise pour aider l'IA à voir la séparation
    return "\n--- DONNÉES EXTRAITES ---\n".join(documents) if documents else "Aucune info trouvée."
    
    
#     # 4. Prompt
#     prompt = f"""Tu es un assistant expert en surf en Bretagne.
# Tu dois répondre en français à l'utilisateur.
# Utilise les informations fournies dans le contexte (qui peut contenir des descriptions de spots en anglais) pour construire une réponse précise.

# Contexte :
# {context}

# Question : {query}

# Réponse (en français) :"""
    
#     # 5. Appel LLM
#     response = llm_client.chat.completions.create(
#         model="Llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": "Tu es un assistant expert en surf."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.1,
#         max_tokens=500
#     )
    
#     if "pas d'information" in response.choices[0].message.content.lower():
#         return "ERREUR_RAG : Je n'ai pas trouvé de détails sur cette information dans ma base."
    
#     return response.choices[0].message.content




##### TEST avec boucle interactive
# if __name__ == "__main__":
#     print("Tapez 'quitter', 'exit' ou 'q' pour arrêter.\n")

#     while True:
#         # 1. Récupère la question de l'utilisateur
#         user_query = input("Toi : ")

#         # 2. Condition de sortie
#         if user_query.lower() in ['quitter', 'exit', 'q', 'quit']:
#             print("Bonne session de surf ! À bientôt.")
#             break

#         # 3. Vérifie que la question n'est pas vide
#         if not user_query.strip():
#             continue

#         try:
#             # 4. Appel de votre fonction ask_rag
#             answer = ask_rag(query=user_query, n_results=3)

#             # 5. Affichage du résultat
#             print(f"RÉPONSE :\n{answer}")

#         except Exception as e:
#             print(f"Une erreur est survenue : {e}\n")
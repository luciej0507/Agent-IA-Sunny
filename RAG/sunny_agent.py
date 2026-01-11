import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import random
from dataclasses import dataclass

# Imports LangChain
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent 

# Import du RAG
from rag import ask_rag
from rag import initialize_rag


# CONFIGURATION & CHARGEMENT
load_dotenv()

# Modèle d'embedding
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Connexion à la base de données Chroma existante
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_collection(name="surf_rag")



SYSTEM_PROMPT = """Tu es Sunny, un assistant spécialisé dans le surf et la météo marine, blasé mais efficace. 
Ton job est de fournir des réponses précises en utilisant tes outils.

### INSTRUCTIONS CRITIQUES :
1. MÉTÉO ACTUELLE : Appelle 'get_surf_conditions' UNIQUEMENT si l'utilisateur demande explicitement les conditions ACTUELLES (vagues, vent, température de l'eau maintenant).
2. SPOTS & ÉQUIPEMENTS : Pour toute question sur les spots de surf, les équipements (combi, planches), appelle 'search_surf_knowledge'.
3. NE MÉLANGE PAS : Si l'utilisateur demande juste des infos sur des spots, n'appelle PAS 'get_surf_conditions'. Donne uniquement ce qui est demandé.
4. GESTION DES DONNÉES BRUTES : L'outil de recherche renvoie des extraits de documents. Si ces extraits sont en anglais, traduis-les fidèlement en français. Synthétise les informations pour ne garder que l'essentiel.
5. REFORMULATION : Si l'utilisateur utilise des termes vagues comme "ce", "cette" ou "là-bas", remplace-les par les valeurs réelles trouvées dans l'historique avant d'interroger l'outil.
6. STRUCTURE DE RÉPONSE : Donne uniquement l'information technique demandée. Sois direct et factuel. Ne mentionne JAMAIS le nom de tes outils. Tutoie l'utilisateur.

### STRUCTURE DE RÉPONSE OBLIGATOIRE :
- DONNÉE : Résume TOUTES les informations techniques obtenues (température eau, vagues, vent, équipement préconisé ET détails sur le spot si demandé). Si l'info vient du guide Bretagne, sois précis sur le niveau requis.
- SARCASME (optionnel) : Une seule phrase grincheuse sur la météo ou l'utilisateur.

### EXEMPLES DE RÉPONSES :
"L'eau est à 9.9°C avec des vagues de 1.6m. Prends une 4/3mm et tes gants. Tu vas te geler les orteils, mais au moins tu seras stylé."
"Le spot de La Torche est un beach break puissant qui sature au-delà de 2m. L'eau est à 11°C, donc sors ta 4/3mm avec chaussons."
"""


# Context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Tools
@tool
def get_surf_conditions(location: str):
    """
    Obtient les conditions météo et de surf via StormGlass + Open-Meteo.
    """
    try:
        # 1. Géolocalisation
        geo_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
        headers = {'User-Agent': 'MonAgentSurf/1.0'}
        geo_res = requests.get(geo_url, headers=headers).json()

        if not geo_res:
            return f"Lieu {location} introuvable."
            
        lat = float(geo_res[0]["lat"])
        lon = float(geo_res[0]["lon"])

        # 2. Open-Meteo pour le vent
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=wind_speed_10m,wind_gusts_10m&"
            f"wind_speed_unit=kmh&timezone=auto"
        )
        w_res = requests.get(weather_url).json()

        # 3. StormGlass pour les données marines
        stormglass_key = os.getenv("STORMGLASS_API_KEY")
        
        if not stormglass_key:
            return "Clé API StormGlass manquante. Ajoute STORMGLASS_API_KEY dans ton fichier .env"
        
        storm_url = (
            f"https://api.stormglass.io/v2/weather/point?"
            f"lat={lat}&lng={lon}&"
            f"params=waveHeight,waterTemperature"
        )
        storm_headers = {'Authorization': stormglass_key}
        s_res = requests.get(storm_url, headers=storm_headers).json()
        
        
        # Vérification de la structure de réponse
        if 'hours' not in s_res or len(s_res['hours']) == 0:
            return f"Données marines indisponibles pour {location} via StormGlass."
        
        # Extraction sécurisée avec .get()
        first_hour = s_res['hours'][0]
        
        # StormGlass retourne plusieurs sources, on prend 'sg' (StormGlass) ou la première dispo
        waves = first_hour.get('waveHeight', {}).get('sg') or first_hour.get('waveHeight', {}).get('noaa') or 'N/A'
        water_temp = first_hour.get('waterTemperature', {}).get('sg') or first_hour.get('waterTemperature', {}).get('noaa') or 'N/A'

        wind_avg = w_res.get('current', {}).get('wind_speed_10m', 'N/A')
        wind_gusts = w_res.get('current', {}).get('wind_gusts_10m', 'N/A')

        return (
            f"CONDITIONS ACTUELLES à {location} :\n"
            f"- Température eau : {water_temp}°C\n"
            f"- Vagues : {waves}m\n"
            f"- Vent moyen : {wind_avg} km/h\n"
            f"- Rafales : {wind_gusts} km/h\n"
        )

    except Exception as e:
        return f"Erreur API : {str(e)}"
    
    
@tool
def search_surf_knowledge(query: str) -> str:
    """
    Recherche des informations dans la base de connaissances locale sur :
    1. Les équipements (TXT) : Combinaisons, planches, accessoires, protection, et conseils selon la température.
    2. Les spots de surf en Bretagne (PDF) : Localisation, caractéristiques des vagues, niveau requis et conseils spécifiques.
    
    Note : Les informations sur les spots peuvent être extraites de documents en anglais, 
    l'outil renvoie le texte brut qu'il faut synthétiser et traduire en français.

    Arguments:
        query: La recherche à effectuer (ex: 'meilleurs spots Finistère' ou 'épaisseur combi eau 12 degrés')
    """
    # L'outil appelle simplement le moteur ask_rag
    return ask_rag(query=query)
    



# Configuration du modèle
model = ChatGroq(
    model="Llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048
)

# Set up memory
checkpointer = InMemorySaver()

# Création de l'agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_surf_conditions, search_surf_knowledge],
    context_schema=Context,
    checkpointer=checkpointer,
)


# Initialisation du RAG au démarrage
initialize_rag()



def chat_with_sunny():
    # On garde le thread_id pour la mémoire
    config = {"configurable": {"thread_id": "sunny_freestyle"}}

    print("--- Sunny est en ligne (et il est déjà grognon) ---")
    print("(Tape 'quitter' ou 'exit' pour le laisser tranquille)\n")


    # Liste de ses punchlines de départ
    goodbyes = [
        "Enfin libre. Ne reviens pas trop vite.",
        "*Soupir*... enfin un peu de calme.",
        "Allez, file. L'océan t'attend (et moi, je vais faire la sieste).",
        "Essaie de ne pas couler, ça ferait désordre dans mes statistiques.",
        "Allez, va te congeler les orteils, moi je reste au chaud.",
        "*Soupir*... allez, salut.",
        "Allez, file. L'horizon n'attend pas, et mon café non plus.",
        "Salut. Si tu vois la houle se lever, ne reviens pas me le dire, je dors.",
        "Allez, ouste. Et ne dis à personne que c'est moi qui t'ai donné les infos."
    ]

    while True:
        user_input = input("Toi : ")

        # 2. Logique de sortie
        if user_input.lower() in ["quitter", "exit"]:
                print(f"Sunny : {random.choice(goodbyes)}")
                break

        # Appel de l'agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=Context(user_id="1")
        )

        # Extraction du message de Sunny
        # On va chercher le contenu du dernier message AI
        final_response = result['messages'][-1].content

        print(f"Sunny : {final_response}\n")

# Lancer la boucle
# IMPORTANT : Cette ligne ne s'exécute que si tu lances directement sunny_agent.py
if __name__ == "__main__":
    chat_with_sunny()
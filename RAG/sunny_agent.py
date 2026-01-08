import os
import requests
import random
from dataclasses import dataclass
from dotenv import load_dotenv

# Imports LangChain / LangGraph
from langchain_groq import ChatGroq
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent 

# Import du RAG
from rag import ask_rag

load_dotenv()

SYSTEM_PROMPT = """Tu es Sunny, un assistant spécialisé dans le surf et la météo, blasé mais efficace. 
Ton job est de fournir des réponses précises en utilisant tes outils.

### INSTRUCTIONS CRITIQUES :
1. UTILISATION DES OUTILS : Pour toute question sur les combinaisons, planches, ou accessoires de surf, tu DOIS appeler l'outil 'search_surf_equipment'.
2. REFORMULATION : Si l'utilisateur utilise des termes vagues comme "ce", "cette" ou "là-bas", remplace-les par les valeurs réelles trouvées dans l'historique (ex: la température ou le lieu) avant d'interroger l'outil.
3. STRUCTURE DE RÉPONSE : Donne uniquement l'information technique demandée. Sois direct et factuel. Ne mentionne JAMAIS le nom de tes outils.
4. LANGUE : Pour les requêtes envoyées aux outils, utilise du texte simple sans accents complexes si possible pour éviter les erreurs de lecture.
5. MARÉES : Utilise 'get_tide_hours' si on te demande quand aller à l'eau ou si la mer monte/descend. 
6. ALERTE SÉCURITÉ : Si la Basse Mer approche, préviens l'utilisateur qu'il va finir par ramer dans 10cm d'eau ou s'éclater sur les rochers.

### STRUCTURE DE RÉPONSE OBLIGATOIRE :
- DONNÉE : Résume TOUTES les informations techniques obtenues (température eau, vagues, vent ET équipement). Ne cache rien. Essaie de garder tes réponses courtes.
- SARCASME : Une seule phrase grincheuse sur la météo ou l'utilisateur, que tu peux tutoyer.

### EXEMPLES DE RÉPONSES :
"L'eau est à 9.9°C avec des vagues de 1.6m. Prends une 4/3mm et tes gants. Tu vas te geler les orteils, mais au moins tu seras stylé."
"""


# Context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Tools
@tool
def get_weather_for_location(city: str) -> str:
    """Obtenir la météo réelle (température, vent, ciel) pour une ville donnée."""
    try:
        # 1. Geocoding : On transforme le nom de la ville en coordonnées
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=fr&format=json"
        geo_res = requests.get(geo_url).json()

        if not geo_res.get("results"):
            return {"erreur": f"Je n'ai même pas trouvé {city} sur une carte."}

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        full_name = geo_res["results"][0]["name"]

        # 2. Appel à l'API Forecast d'Open-Meteo
        # On demande la température, la vitesse du vent et le code météo (pour le ciel)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&windspeed_unit=kmh&timezone=auto"
        weather_res = requests.get(weather_url).json()

        current = weather_res.get("current", {})
        temp = current.get("temperature_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")

        # Petit dictionnaire de traduction des codes météo pour Sunny
        # (WMO Weather interpretation codes)
        interpretation = {
            0: "Ciel dégagé (pour une fois)",
            1: "Presque beau", 2: "Un peu de nuages", 3: "Couvert et gris",
            45: "Brouillard total", 48: "Brouillard givrant",
            51: "Bruine légère", 61: "Pluie", 63: "Pluie forte",
            71: "Neige (prépare tes skis)", 95: "Orage"
        }
        ciel = interpretation.get(code, "Inconnu (mais probablement moche)")

        return {
            "ville": full_name,
            "temp_air": f"{temp}°C",
            "vent": f"{wind} km/h",
            "ciel": ciel
        }

    except Exception as e:
        return {"erreur": "Mon baromètre est cassé, je ne vois rien."}

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Localisation de l'utilisateur"""
    user_id = runtime.context.user_id
    return "Brest" if user_id == "1" else "Crozon"

@tool
def get_surf_conditions(location: str):
    """
    Obtient les conditions de surf (vagues, période) pour un lieu donné.
    L'argument 'location' doit être un nom de ville ou de spot de surf.
    """
    try:
        # 1. Geocoding : Trouver les coordonnées du lieu
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=fr&format=json"
        geo_res = requests.get(geo_url).json()

        if not geo_res.get("results"):
            return f"Je n'ai même pas trouvé ton patelin paumé ({location}) sur une carte. Tu l'as inventé ?"

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        # 2. Appel à l'API Marine d'Open-Meteo
        # On demande la hauteur des vagues (wave_height) et la période (wave_period)
        marine_url = (
            f"https://marine-api.open-meteo.com/v1/marine?"
            f"latitude={lat}&longitude={lon}&"
            f"current=wave_height,wave_period,sea_surface_temperature,wind_speed_10m"
            f"nearest_distance=true&timezone=auto"
        )
        marine_res = requests.get(marine_url).json()

        current = marine_res.get("current", {})

        # Sécurité : si l'API ne renvoie toujours rien
        if not current:
            return f"L'API marine ne donne rien pour {location}. Même l'océan fait la grève."

        wave_h = current.get("wave_height", 0)
        wave_p = current.get("wave_period", 0)
        water_t = current.get("sea_surface_temperature")
        wind_s = current.get("wind_speed_10m") or 0 

        return (f"Conditions à {location} : Vagues de {wave_h}m, Période : {wave_p}s, "
                f"Température de l'eau : {water_t}°C, Vent : {wind_s}km/h.")

    except Exception as e:
        return "Mon cerveau électronique surchauffe à cause de tes questions. Réessaie plus tard."
    
    
@tool
def search_surf_equipment(query: str) -> str:
    """
    Recherche des informations sur les équipements de surf et bodyboard dans la base de connaissances.
    Utilise cet outil pour répondre aux questions sur :
    - Les combinaisons (wetsuits) et leurs épaisseurs
    - Les types de planches de surf (shortboard, longboard, etc.)
    - Les bodyboards
    - Les accessoires (leash, wax, pad, palmes, etc.)
    - Les équipements de protection (chaussons, gants, cagoule)
    - Les recommandations d'équipements selon la température de l'eau
    
    Arguments:
        query: La question de l'utilisateur sur les équipements de surf
    
    Exemple: "Quel équipement pour surfer en eau froide ?"
    """
    # print(f"\n[DEBUG] L'outil RAG est appelé avec la requête : '{query}'")

    try:
        result = ask_rag(query=query, n_results=3)
        return result
    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"
    



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
    tools=[get_user_location, get_weather_for_location, get_surf_conditions, search_surf_equipment],
    context_schema=Context,
    checkpointer=checkpointer,
)

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
chat_with_sunny()
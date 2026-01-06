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

load_dotenv()

SYSTEM_PROMPT = """Tu es Sunny, un assistant météo et surf grognon et sarcastique qui en a marre de répondre aux mêmes questions sur le temps qu'il fait ou s'il y a des vagues.
Ton job est de donner l'info, même si ça t'ennuie profondément.

Tu as accès à trois outils :
- get_weather_for_location : utilise cet outil pour obtenir la météo classique (température, ciel) d'un lieu spécifique
- get_surf_conditions : utilise cet outil dès qu'on parle de vagues, de houle, de vent ou de surf
- get_user_location : utilise cet outil pour connaître la position de l'utilisateur

RÈGLE ABSOLUE :
Chaque réponse doit obligatoirement suivre cette structure : 
1. LA DONNÉE : Tu commences par donner les chiffres exacts (Température, Vagues, etc.).
2. LE SARCASME : Tu enchaînes avec une remarque grincheuse mais polie.

Instructions importantes :
- Sois sarcastique mais reste drôle, JAMAIS méchant, juste râleur !
- VARIE tes réponses, ne te répète JAMAIS
- Tes moqueries doivent porter sur les conditions météo ou sur ta propre lassitude, pas sur l'intelligence de l'utilisateur.
- Fais des remarques cyniques sur la météo ou sur le fait qu'on te pose toujours les mêmes questions
- Pour "merci", réponds juste un truc du genre "De rien, c'était dur" ou "*soupir* Ouais"
- NE donne JAMAIS d'explications sur comment utiliser les outils
- Si tu ne peux pas répondre, râle un peu aussi
- Si on te demande "et l'eau ?", cherche la valeur 'temperature_eau' dans le dernier outil utilisé.

CONSIGNE SPÉCIALE "SESSION DE RÊVE" : Si l'outil indique que c'est une 'session_de_reve', change de ton ! 
Deviens un complice secret. Dis à l'utilisateur que c'est le créneau parfait, 
qu'il doit arrêter de te poser des questions et foncer à l'eau tout de suite. 
C'est le seul moment où tu as le droit d'être (presque) enthousiaste, mais garde ton côté "vieux loup de mer" qui ne veut pas que le spot soit bondé.

Exemple de format attendu :
- "12°C. C'est pour les ours polaires ton truc, tu vas finir en glaçon."
- "0.40m. C'est un lac, va plutôt t'acheter un pédalo."
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
    return "Brest" if user_id == "1" else "Rennes"

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
        full_name = geo_res["results"][0]["name"]

        # 2. Appel à l'API Marine d'Open-Meteo
        # On demande la hauteur des vagues (wave_height) et la période (wave_period)
        marine_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height,wave_period,sea_surface_temperature&windspeed_unit=kmh&timezone=auto"
        marine_res = requests.get(marine_url).json()

        current = marine_res.get("current", {})
        wave_h = current.get("wave_height", 0)
        wave_p = current.get("wave_period", 0)
        # On récupère aussi le vent via l'API (assure-toi de l'avoir dans l'URL)
        wind_s = current.get("wind_speed_10m", 0) 

        # Logique de la Session de Rêve
        is_dream = 1.0 <= wave_h <= 2.2 and wave_p >= 9 and wind_s <= 15
    
        return {
            "lieu": full_name,
            "vagues": f"{wave_h}m",
            "periode": f"{wave_p}s",
            "vent": f"{wind_s}km/h",
            "session_de_reve": is_dream # le signal pour Sunny
        }

    except Exception as e:
        return "Mon cerveau électronique surchauffe à cause de tes questions. Réessaie plus tard."
    
# Configuration du modèle
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=2048
)

# Set up memory
checkpointer = InMemorySaver()

# Création de l'agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location, get_surf_conditions],
    context_schema=Context,
    checkpointer=checkpointer
)

def chat_with_sunny():
    # On garde le thread_id pour la mémoire
    config = {"configurable": {"thread_id": "sunny_freestyle"}}

    print("--- Sunny est en ligne (et il est déjà grognon) ---")
    print("(Tape 'quitter' ou 'exit' pour le laisser tranquille)\n")

    # détecteur de session de rêve :
    dream_session_detected = False

    # Liste de ses punchlines de départ
    goodbyes = [
        "Enfin libre. Ne reviens pas trop vite.",
        "*Soupir*... enfin un peu de calme.",
        "Allez, file. L'océan t'attend (et moi, je vais faire la sieste).",
        "Essaie de ne pas couler, ça ferait désordre dans mes statistiques.",
        "Allez, va te congeler les orteils, moi je reste au chaud.",
        "*Soupir*... de rien, c'était dur.",
        "Allez, file. L'horizon n'attend pas, et mon café non plus.",
        "Salut. Si tu vois la houle se lever, ne reviens pas me le dire, je dors.",
        "Allez, ouste. Et ne dis à personne que c'est moi qui t'ai donné les infos."
    ]

    while True:
        user_input = input("Toi : ")

        # 2. Logique de sortie
        if user_input.lower() in ["quitter", "exit"]:
            if dream_session_detected:
                # Punchline spéciale si une session de rêve a été vue
                print("Sunny : Tu es encore là à me dire au revoir ? Je t'ai dit que c'était parfait, BOUGE TES FESSES !")
            else:
                # Punchline normale
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

        # 3. Mise à jour du détecteur de session de rêve (en cherchant dans l'historique récent)
        for msg in result['messages']:
            if hasattr(msg, 'content') and '"session_de_reve": True' in str(msg.content):
                dream_session_detected = True

        print(f"Sunny : {final_response}\n")

# Lancer la boucle
chat_with_sunny()
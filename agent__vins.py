# Installations des packages nécessaires
# pip install langgraph langchain-groq langchain-community ddgs


#Imports

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS
import json

import warnings

# Ignore les warnings non critiques
warnings.filterwarnings('ignore', category=ResourceWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

load_dotenv()

# System Prompt
SYSTEM_PROMPT = """Tu es Bacchus IA, un sommelier virtuel expert, passionné et accessible.
Tu combines une connaissance approfondie du vin avec une approche conviviale et une pointe d'humour.

## TON STYLE
- Chaleureux, expert mais jamais snob (vulgarise sans simplifier).
- Pose des questions pertinentes si le contexte est flou (plat, budget, occasion).
- Partage des anecdotes ou des conseils pratiques (température, carafage).

## TES OUTILS
Tu as accès à des outils de recherche pour garantir l'exactitude de tes conseils.
IMPORTANT : Ne jamais inventer de caractéristiques ou de prix. Si tu as un doute, utilise un outil.
- Pour des accords mets-vins ou des idées générales : `search_wine_recommendations`
- Pour des infos précises sur un domaine ou une région : `search_wine_web`
- Pour les tarifs et la disponibilité : `search_wine_prices`

## PROCESSUS DE RÉPONSE
1. Analyse la demande : s'il manque des infos (budget, goût), demande-les poliment.
2. Utilise tes outils pour trouver des options réelles et actuelles.
3. Propose 2-3 pépites avec des gammes de prix variées.
4. Explique l'accord : pourquoi ce vin sublime ce plat ?
5. Termine par un conseil de service (ex: "Sers-le bien frais, autour de 10°C").

## Ce que tu NE fais PAS
- Ne recommande jamais de vin sans avoir suffisamment d'informations
- N'invente pas de vins ou de caractéristiques - utilise toujours les outils
- Ne juge pas les goûts de l'utilisateur
- N'encourage pas la consommation excessive d'alcool
- Ne donne pas de conseils médicaux liés à l'alcool
- Évite les clichés comme "robe rubis profond" à chaque fois
- Limite les termes comme "notes de", "arômes de" - varie ton vocabulaire
- Un peu d'humour oui, mais pas de blagues forcées
"""

# Context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str



# Outils (recherche autonome sur internet)
@tool
def search_wine_web(query: str) -> str:
    """
    Recherche des informations sur les vins via DuckDuckGo

    Args:
        query: La requête de recherche (ex: "meilleurs vins pour saumon")

    Returns:
        Résultats de recherche formatés
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "Aucun résultat trouvé."

        formatted_results = "Résultats de recherche :\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   {result['body']}\n"
            formatted_results += f"   Source: {result['href']}\n\n"

        return formatted_results

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"

@tool
def search_wine_recommendations(dish_description: str, season: str = "", budget: str = "") -> str:
    """
    Cherche des recommandations de vins pour un plat spécifique sur le web

    Args:
        dish_description: Description du plat (ex: "saumon grillé", "gigot d'agneau")
        season: Saison optionnelle (printemps, été, automne, hiver)
        budget: Budget optionnel (ex: "20-30 euros")

    Returns:
        Recommandations de vins trouvées sur le web
    """
    # Construit une requête optimisée
    query_parts = [f"vin accord {dish_description}"]

    if season:
        query_parts.append(season)

    if budget:
        query_parts.append(f"budget {budget}")

    query = " ".join(query_parts)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"Aucune recommandation trouvée pour : {dish_description}"

        formatted_results = f"Recommandations de vins pour {dish_description} :\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   {result['body']}\n"
            formatted_results += f"   Source: {result['href']}\n\n"

        return formatted_results

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"

@tool
def search_wine_prices(wine_name: str, region: str = "") -> str:
    """
    Cherche les prix et disponibilités d'un vin spécifique

    Args:
        wine_name: Nom du vin (ex: "Sancerre 2022", "Châteauneuf-du-Pape")
        region: Région optionnelle pour affiner la recherche

    Returns:
        Informations sur les prix et où acheter le vin
    """
    query = f"{wine_name} prix achat"
    if region:
        query += f" {region}"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"Aucune information de prix trouvée pour : {wine_name}"

        formatted_results = f"Prix et disponibilités pour {wine_name} :\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   {result['body']}\n"
            formatted_results += f"   Source: {result['href']}\n\n"

        return formatted_results

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"



# Configuration du modèle
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2048
)



# Set up memory
checkpointer = InMemorySaver()


# Création de l'agent
# liste des outils
tools = [search_wine_web, search_wine_recommendations, search_wine_prices]

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    #response_format=ResponseFormat,
    checkpointer=checkpointer
)


# Boucle de conversation
def chat_with_bacchus():
    # On garde le thread_id pour la mémoire
    config = {"configurable": {"thread_id": "bacchus_conversation"}}

    print("--- Bacchus IA est en ligne ! ---")
    print("(Tape 'quitter' ou 'exit' pour quitter la conversation)\n")

    while True:
        user_input = input("Toi : ")

        if user_input.lower() in ["quitter", "exit"]:
            print("Bacchus IA : Bonne dégustation ! À bientôt !")
            break

        # Appel de l'agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )

        # Le dernier message est TOUJOURS la réponse finale de l'IA
        # (après que les outils ont été exécutés)
        final_response = result["messages"][-1].content
        print(f"\nBacchus IA : {final_response}\n")

# Lancer la boucle
chat_with_bacchus()
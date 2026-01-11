import streamlit as st
from sunny_agent import agent, Context
import time
import random

# Configuration de la page
st.set_page_config(
    page_title="🏄 Sunny - Surf Assistant",
    page_icon="🌊",
    layout="centered"
)

# CSS personnalisé style surf "Fun & Warm"
st.markdown("""
    <style>
    /* Fond bleu clair ensoleillé */
    .stApp {
        background: linear-gradient(180deg, #A0E9FF 0%, #FFFFFF 100%);
    }
    
    /* Container principal */
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    
    /* Titre stylé avec un dégradé orange soleil */
    h1 {
        color: #FF5F1F;
        text-align: center;
        font-size: 3.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 800;
    }
    
    /* Sous-titre */
    .subtitle {
        color: #454545;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Style de base pour TOUTES les bulles de chat */
    [data-testid="stChatMessage"] {
        border-radius: 25px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 2px solid rgba(255,255,255,0.5);
    }

    /* Bulle de l'UTILISATEUR (Orange chaud) */
    [data-testid="stChatMessage"]:has([data-testid="stChatUserAvatar"]) {
        background-color: #FF914D !important;
        color: white !important;
        border-bottom-right-radius: 2px;
    }

    /* Bulle de l'ASSISTANT (Jaune soleil) */
    [data-testid="stChatMessage"]:has([data-testid="stChatAssistantAvatar"]) {
        background-color: #FFFBE6 !important;
        color: #454545 !important;
        border-bottom-left-radius: 2px;
    }

    /* FIX: Suppression de la double bordure de l'input */
    .stChatInput {
        border: 2px solid #FF914D !important; /* On garde la bordure orange */
        border-radius: 35px !important;
        background-color: white !important;
        padding: 5px !important;
    }

    /* On retire la bordure interne par défaut de Streamlit */
    .stChatInput textarea {
        border: none !important;
        box-shadow: none !important;
    }

    /* Sidebar style "Sable" */
    [data-testid="stSidebar"] {
        background-color: #FEF9F3;
    }
    
    /* Bouton personnalisé */
    .stButton>button {
        border-radius: 20px;
        background-color: #FF914D;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF5F1F;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🏄 Sunny")
st.markdown('<p class="subtitle">Ton assistant surf (parfois un peu grognon 😅)</p>', unsafe_allow_html=True)

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = "thread_1"  # ID unique pour la conversation

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Pose ta question sur le surf..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Réponse de l'assistant avec streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Configuration pour l'agent
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": "streamlit_user"
            }
        }
        
        # Streaming de la réponse
        import time
        
        try:
            # Récupération de la réponse complète
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config
            )
            
            # Extraction du message final
            full_response = result['messages'][-1].content
            
            # Affichage caractère par caractère pour simuler le streaming
            displayed_text = ""
            for char in full_response:
                displayed_text += char
                message_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.02)  # Ajuste cette valeur (0.01-0.05) pour la vitesse
            
            # Affichage final sans le curseur
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"Erreur : {str(e)}"
            message_placeholder.markdown(error_msg)
            full_response = error_msg
        
        # Sauvegarde de la réponse
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar avec infos
with st.sidebar:
    st.markdown("### 🌊 À propos")
    st.markdown("""
    **Sunny** peut t'aider avec :
    - 🌡️ Les conditions météo et surf en temps réel
    - 🏄 Des conseils sur les spots de Bretagne
    - 🧥 Quel équipement choisir (combi, planche...)
    - (Tape 'Quitter" pour terminer la conversation)
    """)
    
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"thread_{len(st.session_state.get('thread_id', '')) + 1}"
        st.rerun()
    
    st.markdown("---")
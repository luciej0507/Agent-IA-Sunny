import streamlit as st
import random
from sunny_agent import agent, Context # On importe les cerveaux de Sunny

st.set_page_config(page_title="Sunny Surf", page_icon="🏄‍♂️")

# --- STYLE ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .chat-bubble { border-radius: 15px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- LOGIQUE DE SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dream_detected" not in st.session_state:
    st.session_state.dream_detected = False

# --- INTERFACE ---
st.title("🏄‍♂️ Sunny : L'Expert Grognon")

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input utilisateur
if prompt := st.chat_input("Alors, on veut aller à l'eau ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Appel de ton agent LangGraph
    config = {"configurable": {"thread_id": "streamlit_user"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        context=Context(user_id="1")
    )
    
    final_response = result['messages'][-1].content
    
    
    # Vérification Session de Rêve
    if '"session_de_reve": True' in str(result):
        st.session_state.dream_detected = True

    with st.chat_message("assistant"):
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

# Bouton de sortie personnalisé
if st.sidebar.button("Quitter"):
    if st.session_state.dream_detected:
        st.warning("T'ES ENCORE LÀ ?! File à l'eau !")
    else:
        st.info("Enfin libre. Ne reviens pas trop vite.")
    st.stop()
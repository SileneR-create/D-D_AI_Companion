import streamlit as st

st.set_page_config(page_title="D&D AI Companion", layout="wide")
st.title("D&D AI Companion")
st.markdown("Votre assistant intelligent pour Donjons & Dragons 5e")

page = st.navigation([
    st.Page("Home.py", title="🏠 Accueil"),
    st.Page("pages/dnd_chat.py", title="🐉 D&D Rules"),
    st.Page("pages/gamemaster_chat.py", title="🎲 Gamemaster"),
])

page.run()
import streamlit as st
import joblib
import json
from pathlib import Path


st.set_page_config(page_title="ReviewShield", layout="wide")

@st.cache_resource
@st.cache_resource
@st.cache_resource
@st.cache_resource
def load_assets():
    model_path = Path("models/tfidf_lr_calibrated_train_full.joblib")
    z_path = Path("models/quantile_zones.json")
    m = joblib.load(model_path)
    z = json.loads(z_path.read_text(encoding="utf-8"))
    return m, z
model, zones_data = load_assets()


st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Modules", ["Analyseur NLP (Détection IA)", "Analyseur de Réseau (Graphes)"])

if app_mode == "Analyseur NLP (Détection IA)":
    st.title("ReviewShield : Détection de Faux Avis")
    st.markdown("Ce module utilise ton modèle de régression logistique calibré pour déterminer si un avis est écrit par un humain ou généré par une IA/Bot.")

    t_low = zones_data["thresholds"]["t_low"]
    t_high = zones_data["thresholds"]["t_high"]

    review_text = st.text_area("Collez l'avis à analyser ici :")

    if st.button("Lancer l'analyse"):
        if review_text.strip():
            proba = model.predict_proba([review_text])[0][1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Probabilité Faux/IA", f"{proba:.2%}")
            col2.metric("Seuil Humain (<)", f"{t_low:.2%}")
            col3.metric("Seuil Fake (>)", f"{t_high:.2%}")


            if proba < t_low:
                st.success("green")  # Vert pour les avis authentiques
                st.success(" Décision Automatique : Avis Authentique (Humain)")
            elif proba > t_high:
                st.error("red")  # Rouge pour les avis faux
                st.error(" Décision Automatique : Faux Avis (IA ou Spam)")
            else:
                st.error("color:orange")  # Orange pour la zone d'incertitude
                st.warning(" Zone d'Incertitude : Cet avis nécessite une révision manuelle.")

elif app_mode == "Analyseur de Réseau (Graphes)":
    st.title("Détection de Comportements Coordonnés")
    st.info("Module en cours de développement (via `src/graph/coordination_detector.py`). Ce module servira à détecter les fermes à clics en analysant les connexions entre utilisateurs et commerces.")
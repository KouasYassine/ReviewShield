import joblib
import pandas as pd
from pathlib import Path

def test_model():
    # ON CHARGE LE MODÈLE CALIBRÉ (Le plus puissant)
    model_path = Path("models/tfidf_lr_calibrated_train_full.joblib")

    if not model_path.exists():
        print(f"❌ Fichier introuvable : {model_path}")
        return

    model = joblib.load(model_path)

    # TEXTE IA (Le robot académique)
    ai_text = "The integration of artificial intelligence into modern business ecosystems represents a profound paradigm shift in operational efficiency. By leveraging advanced machine learning algorithms, organizations can achieve unprecedented levels of data-driven decision making. Furthermore, the multifaceted nature of digital transformation necessitates a robust framework for ethical implementation. It is imperative to acknowledge that the synergy between human expertise and automated systems is the cornerstone of future innovation. In conclusion, providing seamless architectural solutions ensures that all stakeholders maintain a competitive advantage in an increasingly complex global landscape."

    # TEXTE HUMAIN (Le style Yelp)
    human_text = "The pizza was actually pretty good but the service was slow. I wouldn't go back unless it's happy hour."

    # Prédiction
    # Note : Le modèle attend une liste ou une Series
    proba_ai = model.predict_proba([ai_text])[0][1]
    proba_human = model.predict_proba([human_text])[0][1]

    print("\n" + "="*30)
    print("🎯 VERDICT FINAL DU MODÈLE")
    print("="*30)
    print(f"Texte IA (Attendu: Haut)    -> {proba_ai:.2%}")
    print(f"Texte Humain (Attendu: Bas) -> {proba_human:.2%}")
    print("="*30)

    if proba_ai > 0.70:
        print("\n✅ BRAVO ! Le modèle fonctionne parfaitement.")
        print("Tu peux lancer Streamlit, il sera précis.")
    else:
        print("\n⚠️ ALERTE : Le modèle est bon (96%) mais le test échoue.")
        print("Cause probable : Le texte de test est trop court ou différemment vectorisé.")

if __name__ == "__main__":
    test_model()
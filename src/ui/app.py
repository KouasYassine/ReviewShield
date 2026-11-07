"""ReviewShield demo: paste a product review, get P(AI-generated) and a decision.

    streamlit run src/ui/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import MODEL_DIR  # noqa: E402
from src.models.review_zone import apply_zone  # noqa: E402
from src.models.scorers import available_scorers, load_scorer  # noqa: E402

PRETTY = {"tfidf_lr": "TF-IDF + logistic regression (fast, explainable)",
          "distilbert": "DistilBERT (fine-tuned Transformer)"}
EXAMPLES = {
    "Human-style review": "Bought this for my daughter's dorm. The lid cracked after two weeks but customer "
                          "service sent a new one, no questions asked. Coffee stays hot for maybe 3 hours, not 12.",
    "Generic generated-style review": "This product is great. I love it. It works well and is very easy to use. "
                                      "I would recommend it to anyone who needs a good product. Great quality.",
}

st.set_page_config(page_title="ReviewShield", page_icon="🛡️", layout="centered")


@st.cache_resource
def get_scorer(name):
    return load_scorer(name)


def get_zone(name):
    path = MODEL_DIR / f"review_zone_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


st.title("🛡️ ReviewShield")
st.write("Detects machine-generated product reviews. Confident cases are decided automatically; "
         "uncertain ones are sent to a human moderator.")

models = available_scorers()
if not models:
    st.error("No trained model found in models/. Run `python -m src.models.train_baseline` first.")
    st.stop()

name = st.selectbox("Detector", models, format_func=lambda n: PRETTY.get(n, n))
example = st.selectbox("Try an example (or write your own below)", ["—"] + list(EXAMPLES))
text = st.text_area("Review text", value=EXAMPLES.get(example, ""), height=160)

if st.button("Analyse", type="primary") and text.strip():
    scorer = get_scorer(name)
    p = float(scorer([text])[0])
    zone = get_zone(name)

    st.metric("Probability the review is AI-generated", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))

    if zone:
        t_low, t_high = zone["thresholds"]["t_low"], zone["thresholds"]["t_high"]
        decision = int(apply_zone([p], t_low, t_high)[0])
        target = zone["target_precision"]
        if decision == 1:
            st.error(f"**Flagged as AI-generated** (score ≥ {t_high:.2f}).")
        elif decision == 0:
            st.success(f"**Cleared as human-written** (score ≤ {t_low:.2f}).")
        else:
            st.warning(f"**Needs human review**: score is between {t_low:.2f} and {t_high:.2f}.")
        test = zone["test"]
        st.caption(f"Thresholds tuned on the validation set so that automatic decisions are ≥ {target:.0%} "
                   f"precise. On the test set: {test['auto_rate']:.0%} of reviews decided automatically, "
                   f"{test['auto_accuracy']:.1%} accuracy on those, {test['review_rate']:.0%} sent to review.")
    else:
        st.info("Decision at threshold 0.5 (run `python -m src.models.review_zone` to enable the review zone): "
                + ("**AI-generated**" if p >= 0.5 else "**human-written**"))

    if hasattr(scorer, "explain"):
        exp = scorer.explain(text)
        col1, col2 = st.columns(2)
        col1.markdown("**Pushes towards AI-generated**")
        col1.write(", ".join(f"`{w}`" for w, _ in exp["towards_generated"]) or "—")
        col2.markdown("**Pushes towards human-written**")
        col2.write(", ".join(f"`{w}`" for w, _ in exp["towards_human"]) or "—")

st.divider()
st.caption("Trained on the Fake Reviews Dataset (40K Amazon reviews, GPT-2-generated vs original). "
           "Detection of text from newer LLMs is harder: see the robustness section of the README.")

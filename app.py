from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"
METRICS_PATH = BASE_DIR / "model_comparison.csv"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📧 Email Spam Classifier")
st.write("Enter a message below to predict whether it is **Spam** or **Ham (legitimate)**.")
st.caption("This educational model classifies message text. It does not inspect real email headers, attachments, or sender reputation.")

if not MODEL_PATH.exists():
    st.error("The trained model was not found. Follow the setup steps: place spam.csv in this folder, then run `python train_model.py`.")
    st.stop()

model = load_model()
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

left, right = st.columns([2, 1])
with left:
    message = st.text_area(
        "Email / message content",
        placeholder="Example: Congratulations! You have won a free prize. Click here to claim...",
        height=180
    )
    if st.button("Classify Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([clean_text(message)])[0]
            if prediction == "spam":
                st.error("🚨 Prediction: SPAM")
                st.write("This message has patterns similar to spam messages in the training dataset.")
            else:
                st.success("✅ Prediction: HAM (Likely legitimate)")
                st.write("The model did not classify this message as spam. This does not guarantee that it is safe.")
            # Display a score only if the classifier supports it.
            try:
                probabilities = model.predict_proba([clean_text(message)])[0]
                classes = list(model.classes_)
                spam_score = float(probabilities[classes.index("spam")])
                st.metric("Estimated spam probability", f"{spam_score:.1%}")
            except (AttributeError, ValueError):
                st.caption("The selected SVM model does not provide calibrated probabilities.")

with right:
    st.subheader("Project details")
    st.metric("Training messages", f"{metadata.get('rows_used', 0):,}")
    st.write("**Selected model:**", metadata.get("best_model", "Not available"))
    st.write("**Text processing:** TF-IDF")
    st.write("**Classes:** Spam / Ham")

st.divider()
st.subheader("Model comparison")
if METRICS_PATH.exists():
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}",
        "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True, hide_index=True)
    chart_data = metrics.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
    st.bar_chart(chart_data)
else:
    st.info("Run train_model.py first to create model comparison results.")

st.subheader("Important spam-indicating keywords")
try:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    weights = None
    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
        # For binary sklearn models, positive weights usually favor classes_[1].
        classes = list(classifier.classes_)
        if classes[1] != "spam":
            weights = -weights
    elif hasattr(classifier, "feature_log_prob_"):
        classes = list(classifier.classes_)
        spam_index = classes.index("spam")
        ham_index = classes.index("ham")
        weights = classifier.feature_log_prob_[spam_index] - classifier.feature_log_prob_[ham_index]

    if weights is not None:
        top_indices = weights.argsort()[-12:][::-1]
        keywords = pd.DataFrame({
            "Keyword / phrase": feature_names[top_indices],
            "Spam indicator score": weights[top_indices]
        })
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(keywords["Keyword / phrase"][::-1], keywords["Spam indicator score"][::-1])
        ax.set_xlabel("Relative model weight (not a guarantee)")
        ax.set_title("Terms associated with spam predictions")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Keyword importance is unavailable for this model.")
except Exception:
    st.info("Keyword visualization could not be generated for this model.")

st.caption("Educational prototype: predictions can be wrong. Do not use this alone to make security decisions.")

import streamlit as st
from src.classifier.pipelines.prediction_pipeline import PredictionPipeline

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ================= LOAD MODEL (CACHE) =================
@st.cache_resource
def load_model():
    return PredictionPipeline()

predictor = load_model()

# ================= CUSTOM CSS =================
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00FFAA;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        background-color: #1E1E1E;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #00FFAA;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to detect Fake vs Real News</div>', unsafe_allow_html=True)

# ================= INPUT =================
user_input = st.text_area("Enter News Content Below:", height=200)

# ================= BUTTON =================
if st.button("🔍 Analyze News"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")

    else:
        with st.spinner("Analyzing... 🤖"):
            result = predictor.predict(user_input)

            # OPTIONAL: confidence score (if model supports)
            try:
                proba = predictor.model.predict_proba(
                    predictor.vectorizer.transform([user_input])
                )[0]

                confidence = round(max(proba) * 100, 2)

            except:
                confidence = None

        # ================= RESULT =================
        st.markdown("---")

        if result == "Fake News":
            st.error(f"🚨 This is FAKE NEWS! ({confidence}% confidence)" if confidence else "🚨 This is FAKE NEWS!")
        else:
            st.success(f"✅ This is REAL NEWS! ({confidence}% confidence)" if confidence else "✅ This is REAL NEWS!")

        # ================= EXTRA =================
        st.markdown("### 🔍 Insight")
        st.info("AI models can make mistakes. Always verify from trusted sources.")

        # ================= BONUS =================
        with st.expander("📄 See Input Text"):
            st.write(user_input)
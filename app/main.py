import streamlit as st
from app.chat_logic import load_rag_pipeline, get_rag_response
from app.utils import translate_to_english

st.set_page_config(page_title="AI Healthcare Assistant", layout="centered")
st.title("🩺 RAG-Powered Multilingual Healthcare Assistant")

qa_pipeline = load_rag_pipeline()

lang = st.selectbox("Choose your language", ["en", "fr", "hi", "es", "de"])
query = st.text_input("Ask a health-related question:")

if st.button("Submit") and query:
    if lang != "en":
        query = translate_to_english(query, src_lang=lang)
    response = get_rag_response(query, qa_pipeline)

    # Filter unsafe outputs
    unsafe_keywords = ["dosage", "prescribe", "take", "pill", "tablet"]
    if any(word in response.lower() for word in unsafe_keywords):
        st.warning("⚠️ Please consult a licensed medical professional.")
    else:
        st.markdown(f"**Answer:** {response}")


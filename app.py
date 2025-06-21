# app.py

import streamlit as st
from microagg import (
    load_pipeline,
    load_kb_embedder_faiss,
    load_rephraser,
    classify_and_explain,
    generate_rephrasing
)

st.set_page_config(page_title="Microaggression Rephraser", layout="centered")
st.title("🧠 Microaggression Detector & Rephraser")

user_input = st.text_area("✍️ Enter your sentence:")

if st.button("🔍 Analyze") and user_input:
    with st.spinner("Classifying and generating response..."):
        pipeline = load_pipeline()
        kb, embedder, index = load_kb_embedder_faiss()
        rephraser_pipeline = load_rephraser()

        label, explanation = classify_and_explain(user_input, pipeline, embedder, kb, index)
        st.subheader(f"Prediction: {label}")

        if label == "Microaggression":
            st.write(f"**Explanation:** {explanation}")
            rephrased = generate_rephrasing(user_input, rephraser_pipeline)
            st.write(f"**Suggested Rephrasing:** {rephrased}")
        else:
            st.success("✅ This sentence is not considered a microaggression.")

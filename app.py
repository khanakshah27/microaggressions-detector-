import streamlit as st
from mgagg import classify_and_explain, generate_rephrasing

st.set_page_config(page_title="Microaggression Detector", layout="centered")

st.title("🧠 Microaggression Classifier & Rephraser")
st.markdown("Detect whether a sentence contains a microaggression and suggest a respectful alternative.")

user_input = st.text_area("Enter a sentence:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        label, explanation = classify_and_explain(user_input)
        st.subheader("Prediction:")
        st.write(f"**{label}**")

        if explanation:
            st.subheader("Explanation:")
            st.write(explanation)

            rephrased = generate_rephrasing(user_input)
            st.subheader("Suggested Rephrasing:")
            st.success(rephrased)
        else:
            st.info("No rephrasing needed.")

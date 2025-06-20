import streamlit as st
from mgagg import classify_and_explain, generate_rephrasing


st.title("Microaggression Classifier & Rephraser")

user_input = st.text_area("Enter a sentence")

if st.button("Analyze"):
    label, explanation = classify_and_explain(user_input)
    st.markdown(f"**Prediction:** {label}")
    
    if explanation:
        st.markdown(f"**Explanation:** {explanation}")
        rephrased = generate_rephrasing(user_input)
        st.markdown(f"**Suggested Rephrasing:** {rephrased}")
    else:
        st.success("No microaggression detected.")

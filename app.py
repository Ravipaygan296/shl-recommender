import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a natural language job description or query to get assessment recommendations.")

# --- Input Section ---
query = st.text_area("Enter Job Description or Natural Language Query")

if st.button("Get Recommendations"):
    if query.strip() == "":
        st.warning("Please enter a query to continue.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post("http://localhost:8000/recommend", json={"query": query})
                if response.status_code == 200:
                    results = response.json()["recommendations"]
                    if results:
                        st.success("Found matching assessments!")
                        for rec in results:
                            with st.expander(rec['assessment_name']):
                                st.markdown(f"**URL:** [{rec['assessment_name']}]({rec['url']})")
                                st.markdown(f"**Remote Testing Support:** {rec['remote_testing']}\n")
                                st.markdown(f"**Adaptive/IRT:** {rec['adaptive_support']}\n")
                                st.markdown(f"**Duration:** {rec['duration']}\n")
                                st.markdown(f"**Test Type:** {rec['test_type']}\n")
                    else:
                        st.info("No relevant assessments found.")
                else:
                    st.error("Failed to get response from API.")
            except Exception as e:
                st.exception(e)

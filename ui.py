import os
import streamlit as st
import requests

# Reads BACKEND_URL env var; falls back to localhost for local dev
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_URL = f"{BACKEND_URL}/predict"

st.set_page_config(page_title="Intel Image Classifier")

st.title("Intel Image Classification")
st.write("Upload an image to classify it into: **buildings, forest, glacier, mountain, sea, or street**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=500)

    if st.button(" Classify Image"):
        with st.spinner("Processing image..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)
                result = response.json()

                if result.get("success", False):
                    st.success(f" **Result:** {result['prediction']}")
                    st.info(f" **Confidence:** {result['confidence']}")

                    if "confidence_score" in result:
                        st.progress(result["confidence_score"] / 100)

                    if "all_probabilities" in result:
                        st.write("### All Class Probabilities")
                        for cls, prob in result["all_probabilities"].items():
                            st.write(f"- **{cls}**: {prob:.2f}%")
                else:
                    st.error(f"**Error:** {result.get('error', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(" **Connection Error:** Cannot connect to API. Make sure the server is running!")
                st.info(" Run the API server with: `uvicorn app:app --reload --host 0.0.0.0 --port 8000`")
            except Exception as e:
                st.error(f"**Unexpected Error:** {str(e)}")

st.sidebar.header("About")
st.sidebar.write("""
This app classifies images into 6 categories:
- 🏢 Buildings
- 🌲 Forest
- 🧊 Glacier
- 🏔️ Mountain
- 🌊 Sea
- 🛣️ Street
""")
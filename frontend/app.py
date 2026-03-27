import streamlit as st
import requests
from PIL import Image
import io
import os

# Get API URL from environment variable, default to local docker network service name
API_BASE_URL = os.getenv("API_URL", "http://api:8000")
API_URL = f"{API_BASE_URL}/predict"

st.set_page_config(page_title="caltech gissar", layout="centered")
st.title("caltech gissar")
st.write("Upload a photo of anything and the model will predict what it is.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        uploaded_file.seek(0)
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")},
        )

    if response.status_code == 200:
        result = response.json()

        st.subheader(f"Prediction: **{result['prediction'].replace('_', ' ').title()}**")
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.caption(f"Processing time: {result['processing_time_ms']}ms")

        # Top 5 probabilities chart
        sorted_probs = sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        breeds = [b.replace("_", " ").title() for b, _ in sorted_probs]
        probs = [p for _, p in sorted_probs]

        st.bar_chart(dict(zip(breeds, probs)))
    else:
        st.error(f"API error: {response.status_code} — {response.text}")

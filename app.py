import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from disease_info import DISEASE_INFO
import os

# =========================================================
# GEMINI AI CONFIGURATION
# =========================================================

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )
else:
    gemini_client = None
def generate_ai_explanation(disease, confidence):

    if gemini_client is None:
        return (
            "Gemini AI is not configured. "
            "Please check the GEMINI_API_KEY environment variable."
        )

    prompt = f"""
You are an agricultural education assistant.

A crop disease prediction model has analyzed a leaf image.

Predicted condition: {disease}
Model confidence: {confidence:.2f}%

Explain this prediction in simple language suitable for a college project.

Include:

1. What the condition means
2. Common symptoms
3. General prevention and management practices
4. A short caution that AI predictions should be confirmed by an agricultural professional

Do not provide pesticide doses or unsafe treatment instructions.
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            f"Unable to generate the GenAI explanation: {e}"
        )

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Crop Disease Prediction",
    page_icon="🌱",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.prediction-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "crop_disease_model.keras"
    )


@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌱 AI Crop Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered system for detecting crop diseases from leaf images'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# INFORMATION
# =========================================================

st.info(
    "Upload a clear image of a crop leaf. "
    "The trained deep-learning model will analyze the image "
    "and predict the most likely condition."
)

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📷 Upload Crop Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Crop Leaf",
        use_column_width=True
    )

    st.write("")

    if st.button(
        "🔍 Analyze Leaf",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing the crop leaf..."
        ):

            # Resize image
            resized_image = image.resize(
                (224, 224)
            )

            # Convert to NumPy
            image_array = np.array(
                resized_image
            )

            # Convert image to NumPy array
            # Model already performs Rescaling(1./127.5, offset=-1)
            image_array = np.array(resized_image).astype("float32")
            
            
            
            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                predictions[0][predicted_index]
                * 100
            )

        # =================================================
        # RESULTS
        # =================================================

        st.success(
            "✅ Analysis Complete!"
        )

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )

        st.subheader("🌿 Prediction Result")

        st.write(
            f"### {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )
        
        # =================================================
        # DISEASE INFORMATION / GENAI-STYLE EXPLANATION
        # =================================================

        st.divider()

        st.subheader("🤖 GenAI Disease Explanation")

        with st.spinner("Generating an AI explanation..."):

            ai_explanation = generate_ai_explanation(
                predicted_class,
                confidence
            )

        st.markdown(ai_explanation)

        info = DISEASE_INFO.get(predicted_class)

        if info:

            st.write(f"**What is it?**")

            st.write(info["description"])

            st.write("**🔎 Common symptoms**")

            st.write(info["symptoms"])

            st.write("**💡 General management**")

            st.write(info["management"])

        else:

            st.write(
                "Detailed information for this prediction "
                "is not currently available."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.divider()

        st.subheader("🤖 Model Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model:** CNN")

        with col2:
            st.write("**Classes:** 15")

        st.write(
            "The model was trained using the "
            "PlantVillage dataset."
        )

        # =================================================
        # DISCLAIMER
        # =================================================

        st.warning(
            "⚠️ This system is intended for educational "
            "and research purposes. Predictions should "
            "not replace professional agricultural advice."
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Crop Disease Prediction System | "
    "Built with Python, TensorFlow & Streamlit"
)
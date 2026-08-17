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


# =========================================================
# GENAI DISEASE EXPLANATION
# =========================================================

def generate_ai_explanation(disease, confidence, language):

    if gemini_client is None:

        if language == "हिन्दी":
            return (
                "Gemini AI कॉन्फ़िगर नहीं किया गया है। "
                "कृपया GEMINI_API_KEY की जाँच करें।"
            )

        return (
            "Gemini AI is not configured. "
            "Please check the GEMINI_API_KEY environment variable."
        )

    prompt = f"""
You are an agricultural education assistant.

A crop disease prediction model has analyzed a leaf image.

Predicted condition: {disease}
Model confidence: {confidence:.2f}%

The user selected this language: {language}

Explain the prediction in simple language suitable for a college project.

If the selected language is English, write the explanation in English.

If the selected language is हिन्दी, write the explanation completely in Hindi
using Devanagari script.

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

        if language == "हिन्दी":
            return (
                "GenAI विवरण अभी उपलब्ध नहीं है। "
                "कृपया कुछ समय बाद दोबारा प्रयास करें।"
            )

        return (
            "GenAI explanation is currently unavailable. "
            "Please try again later."
        )


# =========================================================
# LEAF IMAGE VALIDATION
# =========================================================

def validate_leaf_image(image):
    """
    Perform a simple local check to see whether the uploaded image
    looks like a plant/crop leaf. This does not use Gemini, so the
    disease prediction can still work when the Gemini quota is reached.

    Returns True for images that contain a reasonable amount of
    green/plant-like pixels, otherwise False.
    """

    try:
        # Work with a small copy for a fast local check.
        img = image.convert("RGB").resize((224, 224))
        pixels = np.asarray(img).astype(np.float32) / 255.0

        red = pixels[:, :, 0]
        green = pixels[:, :, 1]
        blue = pixels[:, :, 2]

        # Green pixels are the strongest visual signal for the
        # crop-leaf images used by this project.
        green_mask = (
            (green > red * 1.08) &
            (green > blue * 1.05) &
            (green > 0.18)
        )

        green_ratio = float(np.mean(green_mask))

        # Also check colour variation. This helps reject mostly
        # white/grey images such as shirts, documents and walls.
        max_channel = np.max(pixels, axis=2)
        min_channel = np.min(pixels, axis=2)
        saturation = max_channel - min_channel
        mean_saturation = float(np.mean(saturation))

        # Check that green pixels are not limited to just a tiny
        # corner of the image.
        h, w = green_mask.shape
        center = green_mask[h // 10: 9 * h // 10,
                            w // 10: 9 * w // 10]
        center_green_ratio = float(np.mean(center))

        # A normal crop-leaf photo should contain a noticeable
        # amount of green in or around the centre of the image.
        if green_ratio >= 0.10 and center_green_ratio >= 0.06:
            return True

        # A slightly weaker condition handles some leaf photos with
        # yellowing/damaged areas while still requiring colour variation.
        if green_ratio >= 0.16 and mean_saturation >= 0.12:
            return True

        return False

    except Exception:
        # Never send an image to the disease model if validation fails.
        return False


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

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


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
# LANGUAGE SELECTION
# =========================================================

language = st.selectbox(
    "Language / भाषा",
    ["English", "हिन्दी"]
)


# =========================================================
# PAGE TEXT
# =========================================================

if language == "हिन्दी":

    page_title = "🌱 AI फसल रोग पहचान प्रणाली"

    subtitle = (
        "पौधों की पत्तियों से फसल रोगों की पहचान करने वाली AI प्रणाली"
    )

    info_text = (
        "फसल की पत्ती की एक स्पष्ट तस्वीर अपलोड करें। "
        "प्रशिक्षित डीप-लर्निंग मॉडल तस्वीर का विश्लेषण "
        "करके संभावित रोग की पहचान करेगा।"
    )

    upload_label = "📷 फसल की पत्ती की तस्वीर अपलोड करें"

    analyze_button = "🔍 पत्ती का विश्लेषण करें"

    checking_text = (
        "अपलोड की गई तस्वीर की जाँच की जा रही है..."
    )

    analyzing_text = (
        "फसल की पत्ती का विश्लेषण किया जा रहा है..."
    )

    invalid_image = (
        "❌ यह तस्वीर फसल की पत्ती जैसी नहीं लगती। "
        "कृपया फसल की पत्ती की स्पष्ट तस्वीर अपलोड करें।"
    )

    validation_error = (
        "⚠️ तस्वीर की जाँच अभी उपलब्ध नहीं है। "
        "Gemini AI की सीमा या कनेक्शन की समस्या हो सकती है। "
        "कृपया कुछ समय बाद दोबारा प्रयास करें।"
    )

    prediction_title = "🌿 रोग की पहचान"

    confidence_text = "विश्वास स्तर"

    genai_title = "🤖 GenAI रोग विवरण"

    model_title = "🤖 मॉडल की जानकारी"

    model_text = "मॉडल: CNN"

    classes_text = "कक्षाएँ: 15"

    dataset_text = (
        "यह मॉडल PlantVillage डेटासेट का उपयोग करके प्रशिक्षित किया गया है।"
    )

    disclaimer = (
        "⚠️ यह प्रणाली केवल शैक्षिक और अनुसंधान उद्देश्यों के लिए है। "
        "इसकी भविष्यवाणी पेशेवर कृषि सलाह का विकल्प नहीं है।"
    )

    uploaded_caption = "अपलोड की गई फसल की पत्ती"

    generating_text = (
        "AI विवरण तैयार किया जा रहा है..."
    )

    what_is_it = "यह क्या है?"

    common_symptoms = "🔎 सामान्य लक्षण"

    management = "💡 सामान्य प्रबंधन"

    no_info = (
        "इस भविष्यवाणी के लिए विस्तृत जानकारी अभी उपलब्ध नहीं है।"
    )

    analysis_complete = "✅ विश्लेषण पूरा हुआ!"

else:

    page_title = "🌱 AI Crop Disease Prediction"

    subtitle = (
        "An AI-powered system for detecting crop diseases from leaf images"
    )

    info_text = (
        "Upload a clear image of a crop leaf. "
        "The trained deep-learning model will analyze the image "
        "and predict the most likely condition."
    )

    upload_label = "📷 Upload Crop Leaf Image"

    analyze_button = "🔍 Analyze Leaf"

    checking_text = (
        "Checking the uploaded image..."
    )

    analyzing_text = (
        "Analyzing the crop leaf..."
    )

    invalid_image = (
        "❌ This does not appear to be a crop leaf. "
        "Please upload a clear image of a crop leaf."
    )

    validation_error = (
        "⚠️ The image could not be validated right now. "
        "The Gemini AI quota may have been reached or there may "
        "be a connection problem. Please try again later."
    )

    prediction_title = "🌿 Prediction Result"

    confidence_text = "Confidence"

    genai_title = "🤖 GenAI Disease Explanation"

    model_title = "🤖 Model Information"

    model_text = "Model: CNN"

    classes_text = "Classes: 15"

    dataset_text = (
        "The model was trained using the PlantVillage dataset."
    )

    disclaimer = (
        "⚠️ This system is intended for educational and research purposes. "
        "Predictions should not replace professional agricultural advice."
    )

    uploaded_caption = "Uploaded Crop Leaf"

    generating_text = (
        "Generating an AI explanation..."
    )

    what_is_it = "What is it?"

    common_symptoms = "🔎 Common symptoms"

    management = "💡 General management"

    no_info = (
        "Detailed information for this prediction "
        "is not currently available."
    )

    analysis_complete = "✅ Analysis Complete!"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    f'<div class="main-title">{page_title}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">{subtitle}</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# INFORMATION
# =========================================================

st.info(info_text)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    upload_label,
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption=uploaded_caption,
        use_column_width=True
    )

    st.write("")

    if st.button(
        analyze_button,
        use_container_width=True
    ):

        # =================================================
        # STEP 1: VALIDATE IMAGE
        # =================================================

        with st.spinner(checking_text):

            is_leaf = validate_leaf_image(
                image
            )

        # -------------------------------------------------
        # Image is definitely NOT a leaf
        # -------------------------------------------------

        if is_leaf is False:

            st.error(
                invalid_image
            )

            st.stop()

        # =================================================
        # STEP 2: PREDICT DISEASE
        # =================================================

        with st.spinner(analyzing_text):

            # Resize image
            resized_image = image.resize(
                (224, 224)
            )

            # Convert image to NumPy array
            #
            # The model already performs:
            # Rescaling(1./127.5, offset=-1)

            image_array = np.array(
                resized_image
            ).astype("float32")

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
            analysis_complete
        )

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            prediction_title
        )

        st.write(
            f"### {predicted_class}"
        )

        st.metric(
            confidence_text,
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

        # =================================================
        # GENAI EXPLANATION
        # =================================================

        st.divider()

        st.subheader(
            genai_title
        )

        with st.spinner(
            generating_text
        ):

            ai_explanation = generate_ai_explanation(
                predicted_class,
                confidence,
                language
            )

        st.markdown(
            ai_explanation
        )

        # =================================================
        # DISEASE INFORMATION
        # =================================================

        info = DISEASE_INFO.get(
            predicted_class
        )

        if info:

            st.write(
                f"**{what_is_it}**"
            )

            st.write(
                info["description"]
            )

            st.write(
                f"**{common_symptoms}**"
            )

            st.write(
                info["symptoms"]
            )

            st.write(
                f"**{management}**"
            )

            st.write(
                info["management"]
            )

        else:

            st.write(
                no_info
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.divider()

        st.subheader(
            model_title
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**{model_text.split(':')[0]}:** "
                f"{model_text.split(':')[1].strip()}"
            )

        with col2:

            st.write(
                f"**{classes_text.split(':')[0]}:** "
                f"{classes_text.split(':')[1].strip()}"
            )

        st.write(
            dataset_text
        )

        # =================================================
        # DISCLAIMER
        # =================================================

        st.warning(
            disclaimer
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

if language == "हिन्दी":

    st.caption(
        "AI फसल रोग पहचान प्रणाली | "
        "Python, TensorFlow और Streamlit से बनाया गया"
    )

else:

    st.caption(
        "AI Crop Disease Prediction System | "
        "Built with Python, TensorFlow & Streamlit"
    )
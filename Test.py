import numpy as np
import streamlit as st
import tensorflow as tf
import os
import json
from PIL import Image
from io import BytesIO

# Get the directory where the current Python script is located
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- CORRECTED MODEL PATH DEFINITION ---
model_filename = "plant_disease_prediction_model.h5"
model_path = os.path.join(working_dir, model_filename)


# --- CUSTOMIZATION FUNCTION (Unchanged) ---
def set_custom_styles():
    """Injects custom CSS for a more visually appealing Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* 1. Global Background and Font */
        .stApp {{
            background-color: #f0f8ff; 
            font-family: 'Georgia', serif; 
        }}

        /* 2. Title Styling */
        .stApp h1 {{
            color: #1e8449;
            text-align: center;
            font-size: 2.8em;
            padding: 15px 0 25px 0;
            border-bottom: none;
        }}

        /* 3. Button Styling */
        .stButton>button {{
            background-color: #2ecc71;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s, transform 0.1s;
        }}
        .stButton>button:hover {{
            background-color: #1e8449;
            transform: translateY(-2px);
            color: white;
        }}

        /* 4. Headers (Uploaded Image, Prediction) */
        h2 {{
            color: #2c3e50;
            font-weight: 700;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}

        /* 5. Success Message (Prediction Result) */
        .stSuccess {{
            background-color: #e8f5e9;
            border-left: 6px solid #27ae60;
            color: #1e8449;
            padding: 15px;
            border-radius: 8px;
            font-weight: 600;
        }}

        /* 6. File Uploader Styling (Custom Look) */
        .stFileUploader > div > label > div:nth-child(2) {{
            border: 2px dashed #3498db;
            border-radius: 15px;
            padding: 20px;
            background-color: #ecf0f1;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# --- END CUSTOMIZATION FUNCTION ---


# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model from path: {model_path}")
    st.stop()  # Stop the script if the model can't be loaded

# loading the class names
try:
    with open(os.path.join(working_dir, "class_indices.json"), 'r') as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class_indices.json: {e}")
    st.stop()

# --- NEW FEATURE: loading the disease info ---
try:
    with open(os.path.join(working_dir, "disease_info.json"), 'r') as f:
        disease_data = json.load(f)
except Exception as e:
    st.warning(f"Warning: Could not load disease_info.json. No detailed advice will be available.")
    disease_data = {}  # Set to empty dict if loading fails


# Function to Load and Preprocess the Image using Pillow (unchanged)
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# --- UPDATED: Predict the Class and Confidence Score ---
def predict_image_class(model, image_file, class_indices):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)

    # Get index of the highest prediction
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the confidence score
    confidence = np.max(predictions, axis=1)[0]

    # Map index to name
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown Class")

    return predicted_class_name, confidence


# --- Streamlit App Execution ---

# 1. Apply Custom Styles FIRST
set_custom_styles()

# 2. App Content
st.title('🌿 Plant Disease Classifier')

# 3. Add a Professional Divider
st.markdown("<hr style='border: 1px solid #dcdcdc; margin-top: 0; margin-bottom: 30px;'>", unsafe_allow_html=True)

st.markdown("""
<p style="color:#1e8449; font-size:1.2em; font-weight:bold; text-align:center;">
Welcome! Upload a picture of a plant leaf below to check for potential diseases.
</p>
""", unsafe_allow_html=True)

# File Uploader is placed in a container for better spacing
with st.container():
    uploaded_image = st.file_uploader("Choose a Plant Leaf Image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_bytes = uploaded_image.read()
    image = Image.open(BytesIO(image_bytes))

    # Use a well-defined container for results
    with st.container(border=True):
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.subheader("Uploaded Image")
            resized_img = image.resize((250, 250))
            st.image(resized_img, use_container_width=True)

        with col2:
            st.subheader("Classification Result")

            # Prediction Logic Block
            if st.button('Classify Plant Disease', key='classify_btn'):
                with st.spinner('🔬 Classifying... This may take a moment.'):
                    # --- FEATURE: CONFIDENCE SCORE ---
                    prediction, confidence_score = predict_image_class(model, BytesIO(image_bytes), class_indices)

                    st.success(f'✅ Predicted Disease: **{prediction}** ({confidence_score * 100:.2f}% Confidence)')

                # --- FEATURE: DISEASE INFO & TREATMENT ADVICE ---
                if prediction in disease_data:
                    info = disease_data[prediction]

                    st.markdown("---")

                    with st.expander("🩺 **Treatment and Details**", expanded=True):
                        st.markdown(f"**Symptoms:** {info['symptoms']}")
                        st.markdown(f"**Suggested Treatment:** {info['treatment']}")
                else:
                    st.info("No detailed treatment information available for this class.")

            else:
                st.info("Click 'Classify Plant Disease' to get the prediction.")

            # --- FEATURE: USER FEEDBACK ---
            if 'prediction' in locals():  # Only show feedback after a prediction is made
                st.markdown("---")
                st.markdown("Was this prediction helpful? Your feedback helps us improve! 👇")

                feedback_col1, feedback_col2 = st.columns(2)

                if feedback_col1.button("👍 Yes, correct", key='feedback_yes'):
                    st.toast("Thank you for confirming accuracy!")
                    # Log correct prediction data here

                if feedback_col2.button("👎 No, incorrect", key='feedback_no'):
                    # The text input appears only when "No" is clicked
                    st.session_state['show_correction'] = True

            if 'show_correction' in st.session_state and st.session_state['show_correction']:
                correct_class = st.text_input("Please enter the correct disease name:", key='correction_input')
                if st.button("Submit Correction", key='submit_correction') and correct_class:
                    st.toast(f"Correction for '{prediction}' submitted: {correct_class}")
                    st.session_state['show_correction'] = False  # Hide after submission
                    # Log incorrect prediction and user's correction here

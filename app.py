# app.py

import streamlit as st
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
import base64
from ultralytics import YOLO
import supervision as sv
from groq import Groq
from pytesseract import Output
import numpy as np
import gdown

# -------------------------------
# Model Download and Loading
# -------------------------------

# Define the path where you want the model to be saved relative to your project directory
MODEL_DIR = 'models'  # Define the folder to store the model
FILE_PATH = os.path.join(MODEL_DIR, 'yolov10x_best.pt')  # Modify to store inside 'models' folder
FILE_ID = "15YJAUuHYJQlMm0_rjlC-e_VJPmAvjeiE"  # File ID from the shared link on Google Drive
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def download_model():
    """
    Download the YOLO model from Google Drive if it doesn't exist locally.

    Returns:
        str: Path to the downloaded model file, or None if download fails
    """
    # Check if the model already exists
    if not os.path.exists(FILE_PATH):
        st.info("Downloading YOLOv10x model from Google Drive...")
        # Ensure the directory exists where the model will be saved
        os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
        try:
            # Download the file using gdown
            gdown.download(FILE_URL, FILE_PATH, quiet=False)
            st.success(f"Model downloaded successfully at: {FILE_PATH}")
        except Exception as e:
            st.error(f"Error downloading the model: {e}")
            return None
    else:
        st.info(f"Model already exists at: {FILE_PATH}")

    return FILE_PATH

@st.cache_resource
def load_model():
    """
    Load the YOLO model from the downloaded file.

    Returns:
        YOLO: Loaded YOLO model, or None if loading fails
    """
    model_path = download_model()
    if model_path:
        try:
            model = YOLO(model_path)
            st.success("YOLOv10x model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            return None
    return None

# -------------------------------
# Initialize Groq Client
# -------------------------------

@st.cache_resource
def initialize_groq_client(api_key):
    return Groq(api_key=api_key)

# -------------------------------
# OCR and Image Description Functions
# -------------------------------

def get_image_description(client, image_path):
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
            model="llama-3.2-11b-vision-preview",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting image description: {e}")
        return "Description not available."

def perform_ocr(image, detections, client):
    section_annotations = {}
    for idx, (box, label) in enumerate(zip(detections.xyxy, detections.class_id)):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image[y_min:y_max, x_min:x_max]

        if label != 6:  # Assuming label 6 is 'Picture'
            ocr_result = pytesseract.image_to_string(cropped_image, config='--psm 6', output_type=Output.STRING).strip()
            section_name = {
                0: 'Caption',
                1: 'Footnote',
                2: 'Formula',
                3: 'List-item',
                4: 'Page-footer',
                5: 'Page-header',
                7: 'Section-header',
                8: 'Table',
                9: 'Text',
                10: 'Title'
            }.get(label, 'Unknown')

            if section_name not in section_annotations:
                section_annotations[section_name] = []

            section_annotations[section_name].append(ocr_result)
        else:
            # Handle 'Picture' labels
            temp_image_path = f"temp_image_{idx}.png"
            cv2.imwrite(temp_image_path, cropped_image)
            description = get_image_description(client, temp_image_path)
            os.remove(temp_image_path)

            if 'Picture' not in section_annotations:
                section_annotations['Picture'] = []
            section_annotations['Picture'].append(description)

    return section_annotations

def annotate_image(image, detections):
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image

def process_image(model, image, client):
    results = model(source=image, conf=0.2, iou=0.8)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = annotate_image(image, detections)

    # Optionally display the annotated image in the backend (console)
    # sv.plot_image(annotated_image)

    section_annotations = perform_ocr(image, detections, client)

    return annotated_image, section_annotations

# -------------------------------
# Streamlit UI
# -------------------------------

def main():
    st.set_page_config(page_title="Document Segmentation using YOLOv10x", layout="wide")
    st.markdown("""
        <style>
            body { color: #2a0141; }
            .title { color: blue; font-size: 50px; text-align: center; }
            .subtitle { color: blue; font-size: 40px; text-align: center; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="title">Document Segmentation using YOLOv10x</div>
        <div class="subtitle">Separating documents into different sections and annotating them</div>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()

        # Initialize and load the YOLOv10x model
        model = load_model()

        if model is None:
            st.error("Failed to load the YOLOv10x model. Please try again later.")
            st.stop()

        # Initialize Groq client with your API key
        # Ensure you have set your GROQ_API_KEY in Streamlit's secrets
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except KeyError:
            st.error("GROQ API key not found. Please add it to the Streamlit secrets.")
            st.stop()

        client = initialize_groq_client(api_key=groq_api_key)

        if file_type in ["jpg", "jpeg", "png"]:
            # Process image files
            image = Image.open(uploaded_file).convert("RGB")
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            annotated_image, annotations = process_image(model, image_np, client)

            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)

            st.subheader("Extracted Sections:")
            for section, texts in annotations.items():
                st.markdown(f"**{section}:**")
                for text in texts:
                    st.markdown(f"- {text}")

            # Download annotated image
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            st.download_button(
                label="Download Annotated Image",
                data=img_encoded.tobytes(),
                file_name="annotated_image.jpg",
                mime="image/jpeg"
            )

        elif file_type == "pdf":
            # Process PDF files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
                temp_pdf_path = temp_pdf_file.name
                temp_pdf_file.write(uploaded_file.getvalue())

            try:
                pages = convert_from_path(temp_pdf_path, dpi=300)
                for i, page in enumerate(pages, start=1):
                    st.image(page, caption=f"Page {i}", use_column_width=True)
                    page_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

                    annotated_image, annotations = process_image(model, page_np, client)

                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"Annotated Page {i}", use_column_width=True)

                    st.subheader(f"Extracted Sections from Page {i}:")
                    for section, texts in annotations.items():
                        st.markdown(f"**{section}:**")
                        for text in texts:
                            st.markdown(f"- {text}")

                    # Download annotated image
                    _, img_encoded = cv2.imencode('.jpg', annotated_image)
                    st.download_button(
                        label=f"Download Annotated Image - Page {i}",
                        data=img_encoded.tobytes(),
                        file_name=f"annotated_page_{i}.jpg",
                        mime="image/jpeg"
                    )
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()

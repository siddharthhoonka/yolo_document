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
# Constants and Configuration
# -------------------------------
MODEL_DIR = 'models'
MODEL_FILENAME = 'yolov10x_best.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
FILE_ID = "15YJAUuHYJQlMm0_rjlC-e_VJPmAvjeiE"
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

LABEL_MAP = {
    0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item',
    4: 'Page-footer', 5: 'Page-header', 6: 'Picture', 7: 'Section-header',
    8: 'Table', 9: 'Text', 10: 'Title'
}

# -------------------------------
# Model Download and Loading
# -------------------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            gdown.download(FILE_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error downloading the model: {e}")
            return None
    return MODEL_PATH

@st.cache_resource(hash_funcs={YOLO: lambda _: None})
def load_model():
    model_path = download_model()
    if model_path:
        try:
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading YOLOv10x model: {e}")
    return None

@st.cache_resource(hash_funcs={Groq: lambda _: None})
def initialize_groq_client(api_key):
    return Groq(api_key=api_key)

# -------------------------------
# OCR and Image Processing Functions
# -------------------------------
def get_image_description(client, image_path):
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]}],
            model="llama-3.2-11b-vision-preview",
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Description not available."

def perform_ocr(image, detections, client):
    section_annotations = {}
    for idx, (box, label) in enumerate(zip(detections.xyxy, detections.class_id)):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image[y_min:y_max, x_min:x_max]
        section_name = LABEL_MAP.get(label, 'Unknown')
        if label != 6:
            ocr_text = pytesseract.image_to_string(cropped_image, config='--psm 6', output_type=Output.STRING).strip()
            section_annotations.setdefault(section_name, []).append(ocr_text)
        else:
            temp_image_path = f"temp_image_{idx}.png"
            cv2.imwrite(temp_image_path, cropped_image)
            description = get_image_description(client, temp_image_path)
            os.remove(temp_image_path)
            section_annotations.setdefault('Picture', []).append(description)
    return section_annotations

def annotate_image(image, detections):
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    image = bounding_box_annotator.annotate(scene=image, detections=detections)
    return label_annotator.annotate(scene=image, detections=detections)

def process_image(model, image, client):
    results = model(source=image, conf=0.2, iou=0.8)[0]
    detections = sv.Detections.from_ultralytics(results)
    return annotate_image(image, detections), perform_ocr(image, detections, client)

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Document Segmentation using YOLOv10x", layout="wide")
    st.title("📄 Document Segmentation using YOLOv10x")
    st.markdown("Separating documents into different sections and annotating them")
    
    uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
    
    if uploaded_file:
        model = load_model()
        if model is None:
            st.error("Failed to load the YOLOv10x model.")
            return
        
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except KeyError:
            st.error("API key not found. Please add it to Streamlit secrets.")
            return
        client = initialize_groq_client(api_key=groq_api_key)
        
        for file in uploaded_file:
            file_type = file.name.split('.')[-1].lower()
            st.subheader(f"Processing: {file.name}")
            
            if file_type in ["jpg", "jpeg", "png"]:
                image = Image.open(file).convert("RGB")
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Processing image..."):
                    annotated_image, annotations = process_image(model, image_np, client)
                
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
                st.write(annotations)
            
            elif file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
                    temp_pdf_file.write(file.getvalue())
                    temp_pdf_path = temp_pdf_file.name
                
                pages = convert_from_path(temp_pdf_path, dpi=300, output_folder=tempfile.gettempdir())
                for i, page in enumerate(pages, start=1):
                    st.image(page, caption=f"Page {i}", use_column_width=True)
                    page_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    with st.spinner(f"Processing Page {i}..."):
                        annotated_image, annotations = process_image(model, page_np, client)
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"Annotated Page {i}", use_column_width=True)
                    st.write(annotations)
                os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()

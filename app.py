import streamlit as st
import cv2  # Using opencv-python-headless
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
import base64
from ultralytics import YOLO  # Correct import after installing yolov10
import supervision as sv
from pytesseract import Output
import numpy as np
import gdown

# -------------------------------
# Model Download and Loading
# -------------------------------

MODEL_DIR = 'models'
MODEL_FILENAME = 'yolov10x_best.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
FILE_ID = "15YJAUuHYJQlMm0_rjlC-e_VJPmAvjeiE"  # Replace with your actual File ID
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def download_model():
    """
    Download the YOLO model from Google Drive if it doesn't exist locally.
    """
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv10x model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            gdown.download(FILE_URL, MODEL_PATH, quiet=False)
            st.success(f"Model downloaded successfully at: {MODEL_PATH}")
        except Exception as e:
            st.error(f"Error downloading the model: {e}")
            return None
    else:
        st.info(f"Model already exists at: {MODEL_PATH}")
    return MODEL_PATH

@st.cache_resource
def load_model():
    """
    Load the YOLOv10x model from the downloaded file.
    """
    model_path = download_model()
    if model_path:
        try:
            model = YOLO(model_path)
            st.success("YOLOv10x model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Error loading YOLOv10x model: {e}")
            return None
    return None

# -------------------------------
# OCR and Image Processing Functions
# -------------------------------

def perform_ocr(image, detections):
    section_annotations = {}
    for idx, (box, label) in enumerate(zip(detections.xyxy, detections.class_id)):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image[y_min:y_max, x_min:x_max]

        ocr_result = pytesseract.image_to_string(cropped_image, config='--psm 6', output_type=Output.STRING).strip()
        confidence = pytesseract.image_to_data(cropped_image, config='--psm 6', output_type=Output.DICT)['conf']

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

        section_annotations[section_name].append((ocr_result, max(confidence)))
    
    return section_annotations

def annotate_image(image, detections):
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image

def process_image(model, image):
    results = model(source=image, conf=0.2, iou=0.8)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = annotate_image(image, detections)
    section_annotations = perform_ocr(image, detections)
    return annotated_image, section_annotations

# -------------------------------
# Streamlit UI
# -------------------------------

def main():
    st.set_page_config(page_title="Document Segmentation using YOLOv10x", layout="wide")
    st.title("📄 Document Segmentation using YOLOv10x")
    st.write("Upload multiple images or PDFs for processing.")

    # Multiple File Uploads
    uploaded_files = st.file_uploader("Upload images or PDF files", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        model = load_model()
        if model is None:
            st.error("Failed to load YOLOv10x model.")
            st.stop()

        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            st.subheader(f"Processing: {uploaded_file.name}")

            if file_type in ["jpg", "jpeg", "png"]:
                image = Image.open(uploaded_file).convert("RGB")
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                st.image(image, caption="Uploaded Image", use_column_width=True)
                with st.spinner("Processing image..."):
                    annotated_image, annotations = process_image(model, image_np)

                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)

                st.subheader("Extracted Sections:")
                for section, texts in annotations.items():
                    st.markdown(f"**{section}:**")
                    for text, confidence in texts:
                        st.markdown(f"- {text} (Confidence: {confidence}%)")

            elif file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
                    temp_pdf_path = temp_pdf_file.name
                    temp_pdf_file.write(uploaded_file.getvalue())

                try:
                    pages = convert_from_path(temp_pdf_path, dpi=300)
                    for i, page in enumerate(pages, start=1):
                        st.image(page, caption=f"Page {i}", use_column_width=True)
                        page_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

                        with st.spinner(f"Processing Page {i}..."):
                            annotated_image, annotations = process_image(model, page_np)

                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"Annotated Page {i}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()

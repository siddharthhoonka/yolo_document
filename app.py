import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
import cv2
import fitz 
import torch 
from model_loader import load_model


model = load_model()

if model is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

st.markdown("""
    <style>
        body{
            color: #2a0141;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        <h1 style='color: blue; font-size: 50px;'>Document Segmentation using YOLOv10x</h1>
        <h2 style='color: blue; font-size: 40px;'>Seperating documents into different sections and annotating them</h2>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type in ["jpg", "jpeg", "png"]:
        # Convert the uploaded file to a PIL Image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Perform inference
        results = model(image)

        # Convert the annotated image to a NumPy array
        annotated_image = results[0].plot()

        # Display the annotated image in the Streamlit app
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)

        # Convert annotated image to byte stream for downloading
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', annotated_image_bgr)
        img_bytes = img_encoded.tobytes()

        # Create a download button for the annotated image
        st.download_button(
            label="Download Annotated Image",
            data=img_bytes,
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )

    elif file_type == "pdf":
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name
            temp_pdf_file.write(uploaded_file.getvalue())

        # Open the PDF using PyMuPDF (fitz)
        doc = fitz.open(temp_pdf_path)

        # Process each page
        for i in range(doc.page_count):
            page = doc.load_page(i)  # Load the page

            # Convert the page to a PIL image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Display the page as an image
            st.image(img, caption=f"Page {i+1}", use_container_width=True)

            # Process the page with YOLO detection
            results = model(img)

            # Convert the annotated image to a NumPy array
            annotated_image = results[0].plot()

            # Display the annotated image in the Streamlit app
            st.image(annotated_image, caption=f"Detected Objects - Page {i+1}", use_container_width=True)

            # Convert annotated image to byte stream for downloading
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            _, img_encoded = cv2.imencode('.jpg', annotated_image_bgr)
            img_bytes = img_encoded.tobytes()

            # Create a download button for the annotated image
            st.download_button(
                label=f"Download Image of the page with annotations- Page {i+1}",
                data=img_bytes,
                file_name=f"annotated_image_page_{i+1}.jpg",
                mime="image/jpeg"
            )

        # Clean up the temporary file
        os.remove(temp_pdf_path)

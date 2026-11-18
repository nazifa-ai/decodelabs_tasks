
import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from pytesseract import Output
import pandas as pd

# Configure Tesseract for Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.set_page_config(
    page_title="Intelligent Document Text Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Text Extractor")
st.write(
    "Upload a document image, preprocess it, "
    "and extract text using Optical Character Recognition (OCR)."
)

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image_array = np.array(image)

    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return processed

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    processed_image = preprocess_image(image)

    st.subheader("Image Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.write("Preprocessed Image")
        st.image(
            processed_image,
            use_container_width=True,
            clamp=True
        )

    if st.button("Extract Text", type="primary"):

        with st.spinner("Extracting text..."):

            extracted_text = pytesseract.image_to_string(
                processed_image,
                config="--psm 6"
            )

            ocr_data = pytesseract.image_to_data(
                processed_image,
                output_type=Output.DATAFRAME,
                config="--psm 6"
            )

            ocr_data = ocr_data.dropna(subset=["text"])
            ocr_data["text"] = ocr_data["text"].astype(str).str.strip()

            valid_words = ocr_data[
                (ocr_data["text"] != "") &
                (ocr_data["conf"] >= 0)
            ]

            average_confidence = (
                valid_words["conf"].mean()
                if not valid_words.empty else 0
            )

        st.subheader("Extracted Text")

        if extracted_text.strip():
            st.text_area(
                "Recognized document content",
                extracted_text,
                height=250
            )

            st.download_button(
                "Download Extracted Text",
                data=extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
        else:
            st.warning("No readable text detected. Try a clearer image.")

        st.subheader("OCR Confidence")

        st.metric(
            "Average Word Confidence",
            f"{average_confidence:.2f}%"
        )

        st.caption(
            "OCR confidence indicates the system's confidence "
            "in recognized words. It is not the same as measured accuracy."
        )

        st.subheader("Recognized Word Details")

        if not valid_words.empty:
            st.dataframe(
                valid_words[["text", "conf"]],
                use_container_width=True
            )
        else:
            st.info("No words with confidence scores were found.")

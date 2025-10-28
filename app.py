import time
import joblib as jb
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import cv2 as cv
from tempfile import NamedTemporaryFile
from rembg import remove
from ultralytics import YOLO

# 🩹 Compatibility fix for Python 3.13 (since imghdr was removed)
import sys, types
if sys.version_info >= (3, 13):
    sys.modules['imghdr'] = types.ModuleType('imghdr')

# Streamlit app setup
st.set_page_config(
    page_title="Object Detection V1.0",
    page_icon="✅",
    layout="wide",
)

st.header('Perform Object Detection on Your Image')

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    if cv_image is None:
        st.error("❌ Could not decode the uploaded image. Please upload a valid image file.")
    else:
        st.subheader("Do you want to work with a resized image?")
        choice = st.radio("", ["No", "Yes"])

        if choice == 'Yes':
            height = st.number_input("Height", min_value=100, step=10)
            width = st.number_input("Width", min_value=100, step=10)
            cv_image = cv.resize(cv_image, (int(width), int(height)))
            st.image(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB), channels="RGB", caption="Resized Image")
        else:
            st.image(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB), channels="RGB", caption="Original Image")

        # YOLO model inference
        st.write("🔍 Running YOLO detection...")
        model = YOLO("yolo11n.pt")
        results = model(cv_image)

        # Plot results
        plot = results[0].plot()
        st.image(plot, channels="BGR", caption="YOLO Detection Results")

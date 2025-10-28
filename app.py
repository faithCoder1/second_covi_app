import time  # to simulate a real time data, time loop
import joblib as jb
from PIL import Image
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st
from tempfile import NamedTemporaryFile
from rembg import remove
from ultralytics import YOLO
st.set_page_config(
    page_title="Image processing V1.0", 
    page_icon="✅",
    layout="wide",
)
st.header('Perform Basic Image processing on Your Image') 

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
# Convert the uploaded file to a numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode the image as OpenCV format
    cv_image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    # with st.form("example_form"):
    st.subheader("Do you want to work with a Resized Image?")
    choice = st.radio("", ["No", "Yes"])
    if choice=='Yes':
        height  = st.number_input("height",min_value=100, key="height")
        width = st.number_input("width", min_value=100, key="width")
    # submit = st.form_submit_button("Done")
        cv_image1=cv.resize(cv_image,(height,width))
        st.image(cv.cvtColor(cv_image1, cv.COLOR_BGR2RGB), channels="RGB", caption="Upload Successful") 
        ##work with yolo
        model = YOLO("yolo11n.pt") # load models or custom models

        # predict with the model
        results = model(cv_image1)
        
        plot = results[0].plot()
        st.image(cv.cvtColor(cv_image1, cv.COLOR_BGR2RGB), channels="RGB")
    else:
        model = YOLO("yolo11n.pt") # load models or custom models

        # predict with the model
        results = model(cv_image)
        
        plot = results[0].plot()
        st.image(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB), channels="RGB", caption="Upload Successful") 
        ##work with yolo
        

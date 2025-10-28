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
st.set_page_config(
    page_title="Image processing V1.0", 
    page_icon="✅",
    layout="wide",
)
st.header('Perform Basic Image processing on Your Image') 

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

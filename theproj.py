import streamlit as st
import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
import time

def modelDetection(model_path,image):
    if model_path == "yolov8n.pt":
        model=YOLO('yolov8n')
        res=model(image)
        print(res.names)
        print("the length of the result is",len(res.names))
        return res,"v8"
    elif model_path == "yolov5m_Objects365.pt":
        model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5m_Objects365.pt")
        res=model(image)
        print(model.names)
        print("the length of the result is",len(model.names))
        # print(res)

        return res,"v5"

#write a short description that v5 have 

st.markdown(
    """
    <style>
    .head {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Image Upload and Processing App")

st.markdown('<h1 class="head">Choose an image...</h1>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

#create a dropdown menu with default to yolov8n.pt
st.markdown('<h1 class="head">Select Model...</h1>', unsafe_allow_html=True)
model_path = st.selectbox("", ["yolov8n.pt", "yolov5m_Objects365.pt"])
st.markdown(f'<style>.st-ck select {{ font-size: 18px; }}</style>', unsafe_allow_html=True)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    if st.button('Analyse Image'):
        #calculate time
        start = time.time()
        st.write("Detecting...")
        results,model_version = modelDetection(model_path,image)
        st.write("Detected Components:")
        if model_version == "v8":
            result=results[0]
            for obj in result.boxes.data.tolist():
                _, _, _, _, confidence, cls = obj
                class_name = result.names[cls]
                st.write(f"{class_name} (confidence: {confidence:.2f})")
                end=time.time()
            st.write(f"Time taken using the V8 model: {end-start} seconds")
        else:
            names=results.pandas().xyxy[0].name
            confidence=results.pandas().xyxy[0].confidence
            for i in range(len(names)):
                st.write(f"{names[i]} (confidence: {confidence[i]})")
                end=time.time()
            st.write(f"Time taken using the V5 model: {end-start} seconds")

        

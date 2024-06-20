import streamlit as st
from PIL import Image

import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


# model_path = 'yolov5m_Objects365.pt' 

# model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5m_Objects365.pt")
# img_path = 'images/image1.jpeg'  
# img = Image.open(img_path)
# results = model(img)
# results.show()
# print(results[0].names)
# print(results.pandas().xyxy[0])

# exit()

st.title("Image Upload and Processing App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Process Image'):
        gray_image = image.convert("L")
        st.image(gray_image, caption='Processed Image', use_column_width=True)
        gray_image.save('processed_image.png')
        st.success('Image processed and saved successfully!')

import streamlit as st
import torch
from PIL import Image
#from ultralytics import YOLO
import time

def modelDetection(model_path,image):
    if model_path == "yolov8n.pt":
        model=YOLO('yolov8n')
        res=model(image)
        return res,"v8"
    elif model_path == "yolov5m_Objects365.pt":
        model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5m_Objects365.pt")
        res=model(image)
        return res,"v5"

if __name__ == "__main__":
    st.markdown(
        """
        <style>
        .head {
            font-size: 18px;
        }
        .desc{
            font-size: 12px;
            color: #7c7c7c;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Image Upload and Processing App")

    st.markdown('<h1 class="head">Choose an image...</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])


    st.markdown('<h1 class="head">Select Model...</h1>', unsafe_allow_html=True)
    st.markdown('<h1 class="desc">Yolov5 Objects365.pt has 365 classes and Yolov8n.pt has 80 classes</h1>', unsafe_allow_html=True)
    model_path = st.selectbox("", ["yolov8n.pt", "yolov5m_Objects365.pt"])
    st.markdown(f'<style>.st-ck select {{ font-size: 18px; }}</style>', unsafe_allow_html=True)


    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        
        if st.button('Analyse Image'):
            #calculate time
            start = time.time()
            placeholder = st.empty()

    # Display the detecting message
            placeholder.text("Detecting...")
            results,model_version = modelDetection(model_path,image)
            placeholder.empty()
            
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
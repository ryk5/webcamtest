from ultralytics import YOLO
import numpy as np
import streamlit as st
from PIL import Image
import cv2


# Load YOLO model
model = YOLO("best.pt")  # 👈 Load your trained weights

st.title("YOLO Object Detection with Streamlit")

# Image Upload and Processing
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    results = model(image)
    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Processed Image with Detections", use_column_width=True)

# Video Upload and Processing
uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    tfile = f"temp_video.{uploaded_video.name.split('.')[-1]}"
    with open(tfile, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        stframe.image(annotated_frame, channels="BGR", caption="Processed Video Frame")

    cap.release()

# **Webcam Processing (Fix for Safari on iPhone)**
use_webcam = st.checkbox("Use Webcam for Live Detection")

if use_webcam:
    # Use Streamlit's built-in camera input for mobile compatibility
    st.markdown("**Safari users on iPhone may need to allow camera access manually.**")
    webcam_image = st.camera_input("Take a picture for object detection")

    if webcam_image is not None:
        image = np.array(Image.open(webcam_image))
        results = model(image)
        annotated_image = results[0].plot()

        st.image(annotated_image, caption="Processed Live Webcam Image", use_column_width=True)

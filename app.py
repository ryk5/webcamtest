from ultralytics import YOLO
import numpy as np
import streamlit as st
from PIL import Image
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Load YOLO model
model = YOLO("best.pt")  # 👈 Load your trained YOLO model

st.title("YOLO Object Detection with Streamlit (WebRTC)")

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

# Real-time Webcam Processing using WebRTC (Fix for Safari)
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array (BGR format)
        
        results = model(img)  # Run YOLO detection
        annotated_frame = results[0].plot()  # Draw detections
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.markdown("### 📷 Live Webcam Detection (Safari Compatible)")

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,  # Enables real-time processing
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},  # Enable webcam, disable mic
)

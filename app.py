from ultralytics import YOLO
import numpy as np
import streamlit as st
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Load YOLO model
model = YOLO("best.pt")  # Load your trained YOLO model

st.title("📡 Live Object Detection (iOS Compatible)")

# WebRTC Video Processing
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array
        
        # Run YOLO object detection
        results = model(img)
        annotated_frame = results[0].plot()  # Annotate detected objects
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.markdown("### 📷 Start Live Stream from Your iOS Device")

webrtc_streamer(
    key="live-object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # Optimize real-time processing
)

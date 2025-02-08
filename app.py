# app.py
# iOS-Compatible Real-Time Object Detection with YOLO and Streamlit

from ultralytics import YOLO
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Load YOLO model (ensure 'best.pt' is in your directory)
model = YOLO("best.pt")  # Replace with your trained model

# iOS viewport configuration
st.write("""
<script>
if(/iPad|iPhone|iPod/.test(navigator.userAgent)) {
    document.write('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">');
}
</script>
""", unsafe_allow_html=True)

# App title and description
st.title("📡 Real-Time Object Detection (iOS Compatible)")
st.markdown("""
### 📷 iOS Webcam Instructions:
1. Tap **Start Camera** below
2. Allow camera permissions when prompted
3. Point your device at objects to detect
4. Tap screen to focus (iOS feature)
""")

# Mobile-optimized video processor
class iOSVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Mobile-optimized YOLO detection
        results = model(img, imgsz=320, conf=0.5)  # Reduced input size for performance
        
        # Annotate frame with detections
        annotated_frame = results[0].plot(line_width=1)  # Thinner lines for mobile
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# User activation button (required for iOS)
start_cam = st.button("🚀 Start Camera")

if start_cam:
    # WebRTC configuration
    webrtc_streamer(
        key="ios-object-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=iOSVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
        video_html_attrs={
            "style": {
                "width": "100%",
                "margin": "0 auto",
                "border": "2px solid #e1e4e8",
                "borderRadius": "10px"
            },
            "playsinline": True  # Critical for iOS Safari
        },
    )

# HTTPS reminder and troubleshooting
st.markdown("""
---

### ℹ️ iOS Requirements:
- **Must be accessed via HTTPS** (use Streamlit Cloud or ngrok locally)
- Safari 16.4+ recommended
- Disable Low Power Mode for best performance

### 🔧 Troubleshooting:
1. Refresh page if camera freezes
2. Check Safari Settings > Privacy > Camera Access
3. Force-quit Safari if permissions stuck
4. Ensure device is not in Silent Mode
""")

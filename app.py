from ultralytics import YOLO
import numpy as np
import streamlit as st
import cv2
import pyaudio
import wave
import whisper
import tempfile
import os
from openai import OpenAI  # Import the new OpenAI client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load YOLO model
model = YOLO("best.pt")  # 👈 Load a pretrained model (e.g., YOLOv8 Nano)

# Load Whisper model
whisper_model = whisper.load_model("base")  # Load Whisper base model (smallest and fastest)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Load API key from .env file

# List of COCO dataset nouns (80 categories)
COCO_NOUNS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Function to extract the most relevant COCO noun using OpenAI API
def extract_most_relevant_noun(text):
    # Step 1: Extract the most important noun from the sentence
    step1_prompt = f"""
    Analyze the following sentence and extract the most important noun:
    "{text}"

    Return only the noun. Do not include any additional text or explanation.
    """
    step1_response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 👈 Use GPT-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts the most important noun from a sentence."},
            {"role": "user", "content": step1_prompt}
        ],
        max_tokens=10,
        temperature=0.2
    )
    extracted_noun = step1_response.choices[0].message.content.strip()

    # Step 2: Map the extracted noun to the most relevant COCO noun
    step2_prompt = f"""
    The following is a list of nouns from the COCO dataset:
    {', '.join(COCO_NOUNS)}

    Find the most relevant noun from the list that matches:
    "{extracted_noun}"

    Return only the noun. Do not include any additional text or explanation.
    """
    step2_response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 👈 Use GPT-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant that maps a noun to the most relevant COCO dataset noun."},
            {"role": "user", "content": step2_prompt}
        ],
        max_tokens=10,
        temperature=0.2
    )
    relevant_noun = step2_response.choices[0].message.content.strip()

    return relevant_noun

# Initialize session state for the relevant noun
if "relevant_noun" not in st.session_state:
    st.session_state.relevant_noun = ""

st.title("YOLO Object Detection with Streamlit")

# Webcam Processing with Selection
use_webcam = st.checkbox("Use Webcam for Live Detection")

if use_webcam:
    # Select webcam index
    webcam_index = st.selectbox("Select Webcam", options=[0, 1, 2, 3], index=0, help="Choose the webcam device (0 is the default).")

    cap = cv2.VideoCapture(webcam_index)
    stframe = st.empty()

    if not cap.isOpened():
        st.error(f"Unable to open webcam {webcam_index}. Try selecting another one.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model(frame)

            # Filter results to only show the relevant noun
            if st.session_state.relevant_noun:
                filtered_results = [r for r in results[0].boxes if model.names[int(r.cls)] == st.session_state.relevant_noun]
                results[0].boxes = filtered_results

            # Plot the annotated frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            stframe.image(annotated_frame, channels="BGR", caption=f"Live Webcam Detection (Camera {webcam_index})")

        cap.release()

# Microphone and Whisper Integration
st.markdown("---")
st.header("Microphone Input with Whisper")

# Custom CSS to make the button super big
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        font-size: 30px;
        height: 100px;
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("🎤 Press to Speak", key="big_button"):
    st.write("Listening... Speak now!")

    # Record audio using PyAudio
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Whisper expects 16kHz audio
    CHUNK = 1024
    RECORD_SECONDS = 5

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav_filename = tmpfile.name
        with wave.open(wav_filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

    # Transcribe audio with Whisper
    result = whisper_model.transcribe(wav_filename)
    transcription = result["text"]

    st.write("You said:")
    st.write(transcription)

    # Extract the most relevant COCO noun using OpenAI API
    relevant_noun = extract_most_relevant_noun(transcription)
    st.session_state.relevant_noun = relevant_noun
    st.write(f"The most relevant COCO noun is: **{relevant_noun}**")

    # Remove temp file
    os.remove(wav_filename)

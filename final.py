import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Global variables
cap = None
running = False

# Function to start the camera stream
def start_camera():
    global cap, running
    if running:
        st.warning("Camera is already running!")
        return

    # Use the system's camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    running = True

    process_stream()  # Process the camera stream

# Function to process the video stream or uploaded video
def process_stream():
    global running
    count = 0
    stframe = st.empty()  # Placeholder for displaying frames

    while running:
        ret, img = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        count += 1
        if count % 3 != 0:  # Skip frames to process every third frame
            continue

        # Get frame dimensions
        h, w, _ = img.shape
        results = model.predict(img)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))  # Resize the segmentation mask
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert the frame to RGB format for display in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(img_rgb)

        # Display the frame
        stframe.image(frame_pil, caption="Real-time Pothole Detection", use_column_width=True)

# Function to stop the camera stream
def stop_camera():
    global running, cap
    if not running:
        st.warning("Camera is not running!")
        return

    running = False
    if cap:
        cap.release()

# Streamlit Web Interface
# Loading Image using PIL
im = Image.open("pothole.png")
# Adding Image to web app
st.set_page_config(page_title="Pothole Detection", page_icon=im)

st.title("Real-time Pothole Detection")

st.write("Click on 'Start Camera' to detect potholes in real-time using your system's camera and machine learning with YOLO.")

# Start/Stop buttons with unique keys
col1, col2 = st.columns([4, 1])

with col1:
    if st.button("Start Camera", key="start_button"):
        uploaded_video = None  # Reset uploaded_video when starting the camera
        start_camera()

with col2:
    if st.button("Stop Camera", key="stop_button"):
        stop_camera()

st.write("Or upload a video to analyze for potholes.")
# Video upload functionality
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    # Convert the uploaded file to a format that OpenCV can read
    tfile = uploaded_video.read()  # Read the uploaded file
    with open("temp_video.mp4", "wb") as f:
        f.write(tfile)  # Write it to a temporary file

    # Process the uploaded video
    video_capture = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()  # Placeholder for displaying frames
    count = 0

    while True:
        ret, img = video_capture.read()
        if not ret:
            st.warning("Failed to grab frame from the video or video ended.")
            break

        count += 1
        if count % 3 != 0:  # Skip frames to process every third frame
            continue

        # Get frame dimensions
        h, w, _ = img.shape
        results = model.predict(img)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))  # Resize the segmentation mask
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert the frame to RGB format for display in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(img_rgb)

        # Display the frame
        stframe.image(frame_pil, caption="Pothole Detection from Uploaded Video", use_column_width=True)

    video_capture.release()  # Release the video capture
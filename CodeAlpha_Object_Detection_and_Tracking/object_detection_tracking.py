import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from sort import Sort, iou_batch
import base64

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🟩",
    layout="centered"
)


st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
st.title("Object Detection & Tracking (YOLOv8 + SORT)")

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("pic.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load YOLO model once (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-downloads pretrained weights on first run

model = load_model()

# ----------------------------
# Source selection (webcam / video file)
# ----------------------------
source_option = st.radio("Choose the video source:", ["Webcam", "Video File"])

video_path = None
if source_option == "Video File":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

# ----------------------------
# Helper: recover class label for a tracked box
# (SORT only tracks geometry, so we match it back to the
#  closest YOLO detection from this frame to grab its class name)
# ----------------------------
def get_label(track_box, dets_with_labels):
    if not dets_with_labels:
        return "object"
    trk = np.array([track_box])
    best_iou, best_label = 0.0, "object"
    for (x1, y1, x2, y2, name) in dets_with_labels:
        det = np.array([[x1, y1, x2, y2]])
        iou = iou_batch(trk, det)[0][0]
        if iou > best_iou:
            best_iou, best_label = iou, name
    return best_label

# ----------------------------
# Main real-time loop
# (checkbox pattern: unchecking it cancels the running script/loop,
#  which is how Streamlit lets you "stop" a live video loop)
# ----------------------------
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])

if run:
    if source_option == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if video_path is None:
            st.warning("Please upload the first video")
            st.stop()
        cap = cv2.VideoCapture(video_path)

    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("The video is over.")
            break

        # STEP 1: run YOLO detection on this frame
        results = model(frame, verbose=False)[0]

        detections = []
        dets_with_labels = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_name = model.names[int(box.cls[0])]
            detections.append([x1, y1, x2, y2, conf])
            dets_with_labels.append((x1, y1, x2, y2, cls_name))

        dets_np = np.array(detections) if detections else np.empty((0, 5))

        # STEP 2: update SORT tracker -> persistent IDs across frames
        tracked_objects = tracker.update(dets_np)

        # STEP 3: draw boxes + class label + tracking ID
        for x1, y1, x2, y2, track_id in tracked_objects:
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            label = get_label((x1, y1, x2, y2), dets_with_labels)
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{int(track_id)}", (x1i, max(y1i - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

    cap.release()
else:
    st.write("Click Run to start the video/camera.")
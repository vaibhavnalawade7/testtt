import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time
import sys

print("USING GSTREAMER CAMERA PIPELINE (FINAL)")

# ------------------ PATHS ------------------
MODEL_PATH = "yolov8n.pt"

# ------------------ TTS INIT ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 235)
engine.setProperty('volume', 1.0)
engine.say("System activated")
engine.runAndWait()

# ------------------ SPEECH CONTROL ------------------
last_spoken = {}
last_distances = {}
speech_cooldown = 5
queue = Queue()

# ------------------ OBJECT WIDTH RATIOS ------------------
class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}

# ------------------ SPEECH THREAD ------------------
def speak(q):
    while True:
        if not q.empty():
            label, distance, position = q.get()
            now = time.time()

            if label in last_spoken and now - last_spoken[label] < speech_cooldown:
                continue

            prev = last_distances.get(label)

            if prev:
                if distance < prev - 0.3:
                    motion = "approaching"
                elif distance > prev + 0.3:
                    motion = "going away"
                else:
                    motion = "ahead"
            else:
                motion = "ahead"

            if distance <= 2:
                motion = "very close"

            last_distances[label] = distance

            engine.say(f"{label} is {distance} meters on your {position}, {motion}")
            engine.runAndWait()

            last_spoken[label] = now

            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)

Thread(target=speak, args=(queue,), daemon=True).start()

# ------------------ DISTANCE CALC ------------------
def calculate_distance(box, frame_width, label):
    obj_width = box.xyxy[0][2] - box.xyxy[0][0]

    if label in class_avg_sizes:
        obj_width *= class_avg_sizes[label]["width_ratio"]

    distance = (frame_width * 0.5) / np.tan(np.radians(35)) / (obj_width + 1e-6)
    return round(float(distance), 2)

# ------------------ POSITION ------------------
def get_position(frame_width, coords):
    x1 = coords[0]
    if x1 < frame_width // 3:
        return "left"
    elif x1 < 2 * frame_width // 3:
        return "center"
    else:
        return "right"

# ------------------ LOAD MODEL ------------------
model = YOLO(MODEL_PATH)

# ------------------ CAMERA (OPEN-CV SAFE GSTREAMER) ------------------
gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true sync=false"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: OpenCV cannot open GStreamer pipeline")
    sys.exit(1)

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    results = model(frame, conf=0.4, verbose=False)[0]

    nearest = None
    min_dist = float('inf')

    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        coords = list(map(int, box.xyxy[0]))
        dist = calculate_distance(box, frame.shape[1], label)

        if dist < min_dist:
            min_dist = dist
            nearest = (label, dist, coords)

    if nearest and nearest[1] <= 12:
        pos = get_position(frame.shape[1], nearest[2])
        queue.put((nearest[0], nearest[1], pos))

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()

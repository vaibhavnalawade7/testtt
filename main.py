import cv2
import time
import numpy as np
import pyttsx3
from queue import Queue
from threading import Thread
from ultralytics import YOLO
import sys

print("USING GSTREAMER CAMERA PIPELINE (LOCKED TO /dev/video0)")

# ------------------ MODEL ------------------
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# ------------------ TEXT TO SPEECH ------------------
engine = pyttsx3.init()
engine.setProperty("rate", 220)
engine.setProperty("volume", 1.0)
engine.say("System activated")
engine.runAndWait()

# ------------------ SPEECH CONTROL ------------------
speech_queue = Queue()
last_spoken = {}
last_distance = {}
SPEECH_COOLDOWN = 5

# ------------------ OBJECT WIDTH RATIOS ------------------
class_avg_sizes = {
    "person": 2.5,
    "car": 0.37,
    "bicycle": 2.3,
    "motorcycle": 2.4,
    "bus": 0.3,
    "traffic light": 2.95,
    "stop sign": 2.55,
    "bench": 1.6,
    "cat": 1.9,
    "dog": 1.5,
}

# ------------------ SPEECH THREAD ------------------
def speak_worker(q):
    while True:
        if not q.empty():
            label, distance, position = q.get()
            now = time.time()

            if label in last_spoken and now - last_spoken[label] < SPEECH_COOLDOWN:
                continue

            prev = last_distance.get(label)
            motion = "ahead"

            if prev:
                if distance < prev - 0.3:
                    motion = "approaching"
                elif distance > prev + 0.3:
                    motion = "moving away"

            if distance <= 2:
                motion = "very close"

            last_distance[label] = distance
            last_spoken[label] = now

            engine.say(f"{label} is {distance} meters on your {position}, {motion}")
            engine.runAndWait()

            with q.mutex:
                q.queue.clear()

        time.sleep(0.1)

Thread(target=speak_worker, args=(speech_queue,), daemon=True).start()

# ------------------ DISTANCE ------------------
def calculate_distance(box, frame_width, label):
    obj_width = box.xyxy[0][2] - box.xyxy[0][0]
    if label in class_avg_sizes:
        obj_width *= class_avg_sizes[label]
    distance = (frame_width * 0.5) / np.tan(np.radians(35)) / (obj_width + 1e-6)
    return round(float(distance), 2)

# ------------------ POSITION ------------------
def get_position(frame_width, x1):
    if x1 < frame_width // 3:
        return "left"
    elif x1 < 2 * frame_width // 3:
        return "center"
    else:
        return "right"

# ------------------ CAMERA (PROVEN PIPELINE) ------------------
gst_pipeline = (
    "v4l2src device=/dev/video0 io-mode=2 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Cannot open camera via GStreamer")
    sys.exit(1)

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    results = model(frame, conf=0.4, verbose=False)[0]

    nearest = None
    min_dist = float("inf")

    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dist = calculate_distance(box, frame.shape[1], label)

        if dist < min_dist:
            min_dist = dist
            nearest = (label, dist, x1)

    if nearest and min_dist <= 12:
        label, dist, x1 = nearest
        pos = get_position(frame.shape[1], x1)
        speech_queue.put((label, dist, pos))

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import time
import pyttsx3
import cv2
from ultralytics import YOLO
from threading import Thread
from queue import Queue

# ---------------- INIT ----------------
Gst.init(None)

print("USING PURE GSTREAMER CAMERA (NO OPENCV CAPTURE)")

# ---------------- YOLO ----------------
model = YOLO("yolov8n.pt")

# ---------------- TTS ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 220)
engine.say("System activated")
engine.runAndWait()

speech_queue = Queue()
last_spoken = {}
SPEECH_COOLDOWN = 5

# ---------------- SPEECH THREAD ----------------
def speak_worker(q):
    while True:
        if not q.empty():
            label, dist, pos = q.get()
            engine.say(f"{label} {dist} meters {pos}")
            engine.runAndWait()
            time.sleep(SPEECH_COOLDOWN)

Thread(target=speak_worker, args=(speech_queue,), daemon=True).start()

# ---------------- DISTANCE ----------------
def estimate_distance(box, frame_width):
    obj_width = box.xyxy[0][2] - box.xyxy[0][0]
    return round((frame_width * 0.5) / (obj_width + 1e-6), 2)

def position(frame_width, x):
    if x < frame_width // 3:
        return "left"
    elif x < 2 * frame_width // 3:
        return "center"
    else:
        return "right"

# ---------------- GSTREAMER PIPELINE ----------------
pipeline = Gst.parse_launch(
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true"
)

appsink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

# ---------------- MAIN LOOP ----------------
while True:
    sample = appsink.emit("pull-sample")
    if sample is None:
        continue

    buf = sample.get_buffer()
    caps = sample.get_caps()
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
    frame = frame.reshape((height, width, 3))
    buf.unmap(mapinfo)

    # ---------------- YOLO ----------------
    results = model(frame, conf=0.4, verbose=False)[0]

    nearest = None
    min_dist = float("inf")

    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        dist = estimate_distance(box, width)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if dist < min_dist:
            min_dist = dist
            nearest = (label, dist, x1)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {dist}m",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    if nearest and min_dist <= 12:
        label, dist, x = nearest
        pos = position(width, x)
        speech_queue.put((label, dist, pos))

    # ---------------- DISPLAY WINDOW ----------------
    cv2.imshow("Blind Assistant Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()

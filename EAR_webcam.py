import time
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

MODEL_PATH = "face_landmarker.task"
EAR_THRESHOLD = 0.20
BLINK_MAX_DURATION = 0.40
DROWSINESS_DURATION = 1.00
MICROSLEEP_DURATION = 1.50
# Define the indices for the left and right eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)

# EAR calculation function
def calculate_ear(landmarks, eye_indices, width, height):
    p = [np.array([landmarks[i].x * width, landmarks[i].y * height]) for i in eye_indices]
    A = distance.euclidean(p[1], p[5])
    B = distance.euclidean(p[2], p[4])
    C = distance.euclidean(p[0], p[3])
    return float((A + B) / (2.0 * C)) if C else 0.0

def get_status(duration):
    if duration < BLINK_MAX_DURATION:
        return "BLINKING"
    if duration < DROWSINESS_DURATION:
        return "PROLONGED CLOSURE"
    if duration < MICROSLEEP_DURATION:
        return "DROWSINESS"
    return "MICROSLEEP ALERT"

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("ERROR: Cannot access camera.")

eye_closed_start = None

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw face landmarks and calculate EAR
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            eyes_closed = (
                calculate_ear(lm, LEFT_EYE, w, h) < EAR_THRESHOLD
                and calculate_ear(lm, RIGHT_EYE, w, h) < EAR_THRESHOLD
            )
            now = time.monotonic()

            if eyes_closed:
                eye_closed_start = eye_closed_start or now
                closure_duration = now - eye_closed_start
                status = get_status(closure_duration)
                eye_state = "CLOSED"
            else:
                eye_closed_start = None
                closure_duration = 0.0
                status = "NORMAL"
                eye_state = "OPEN"

            cv2.putText(frame, f"Eye State: {eye_state}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) 
                        if eye_state == "OPEN" else (0, 0, 255), 2)
            
            cv2.putText(frame, f"Duration: {closure_duration:.2f}s", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            cv2.putText(frame, f"Status: {status}", (20, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) 
                        if status == "MICROSLEEP ALERT" else (0, 255, 0), 2)
            
            print(f"State: {eye_state} | Duration: {closure_duration:.2f}s | Status: {status}")
        else:
            eye_closed_start = None
            cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("SafeDrive - MediaPipe EAR Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
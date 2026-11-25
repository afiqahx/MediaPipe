import cv2
import mediapipe as mp
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)

EAR_THRESHOLD = 0.20
BLINK_MAX_DURATION = 0.40
DROWSINESS_DURATION = 1.00
MICROSLEEP_DURATION = 1.50

# MediaPipe eye landmark indices
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]

#EAR calculation formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
def calculate_ear(landmarks, eye_indices, width, height):
    eye = []
    for index in eye_indices:

        landmark = landmarks[index]
        x = landmark.x * width
        y = landmark.y * height

        eye.append(np.array([x, y]))

    # Vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])

    if C == 0:
        return 0.0

    # EAR formula
    ear = (A + B) / (2.0 * C)

    return float(ear)


camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Cannot access camera.")
    exit()

print("Camera connected successfully!")
print("Press Q to exit.")

# eye_closed_start is used to track the time when eyes are first detected as closed
eye_closed_start = None
eye_state = "OPEN"

# face landmarker is used to detect facial landmarks in the video frames captured from the webcam
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:

            print("Failed to capture frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp in milliseconds
        timestamp_ms = time.monotonic_ns() // 1_000_000

        # Detect facial landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Check if any face landmarks were detected
        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            # Calculate EAR
            left_ear = calculate_ear(
                face_landmarks,
                LEFT_EYE,
                width,
                height
            )

            right_ear = calculate_ear(
                face_landmarks,
                RIGHT_EYE,
                width,
                height
            )

            # Average both eyes
            average_ear = (left_ear + right_ear) / 2.0


            # Require both eyes to appear closed
            eyes_closed = (
                left_ear < EAR_THRESHOLD
                and right_ear < EAR_THRESHOLD
            )

            current_time = time.monotonic()

            if eyes_closed:

                if eye_closed_start is None:
                    eye_closed_start = current_time
                closure_duration = (
                    current_time - eye_closed_start
                )

                eye_state = "CLOSED"

            else:

                # Reset closure timer
                eye_closed_start = None
                closure_duration = 0.0
                eye_state = "OPEN"

            # Blink and drowsiness detection logic
            status = "NORMAL"

            if eye_state == "CLOSED":
                if closure_duration < BLINK_MAX_DURATION:
                    status = "BLINKING"

                elif closure_duration < DROWSINESS_DURATION:
                    status = "PROLONGED CLOSURE"

                elif closure_duration < MICROSLEEP_DURATION:
                    status = "DROWSINESS"

                else:
                    status = "MICROSLEEP ALERT"

            # display results
            cv2.putText(
                frame,
                f"Eye State: {eye_state}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if eye_state == "OPEN" else (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Duration: {closure_duration:.2f}s",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if status == "MICROSLEEP ALERT" else (0, 255, 0),
                2
            )

            print(
                f"State: {eye_state} | "
                f"Duration: {closure_duration:.2f}s | "
                f"Status: {status}"
            )

        else:

            # No face detected
            eye_closed_start = None

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        #display the webcam 
        cv2.imshow(
            "SafeDrive - MediaPipe EAR Detection",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

print("SafeDrive detection stopped.")

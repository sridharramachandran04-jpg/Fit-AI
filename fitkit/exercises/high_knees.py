import cv2, mediapipe as mp, numpy as np, time, os, tempfile
from PIL import Image
import streamlit as st
from collections import deque
from realtime_feedback import draw_feedback_overlay, render_dashboard, speak_js, FeedbackThrottler
import os as _os

# Paths
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_APP_DIR = _os.path.dirname(_BASE_DIR)

# Mediapipe setup
confidence_threshold = 0.5


# ✅ Angle calculation (kept for consistency)

def _mp():
    """Lazy mediapipe accessor — ensures mp.solutions is patched before use."""
    import mediapipe as mp
    return mp.solutions

def _mp_pose():
    return _mp().pose

def _mp_drawing():
    return _mp().drawing_utils

def _drawing_spec(**kw):
    return _mp_drawing().DrawingSpec(**kw)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    return np.degrees(
        np.arccos(
            np.clip(
                np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)),
                -1, 1
            )
        )
    )


# ✅ REQUIRED FUNCTION (this fixes your error)
def high_knees_detection():
    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None

    with _mp_pose().Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Right side
                hip_y = landmarks[_mp_pose().PoseLandmark.RIGHT_HIP.value].y
                knee_y = landmarks[_mp_pose().PoseLandmark.RIGHT_KNEE.value].y

                # High knees logic
                if knee_y < hip_y:
                    stage = "up"

                if knee_y > hip_y and stage == "up":
                    stage = "down"
                    counter += 1

            except:
                pass

            # Draw skeleton
            _mp_drawing().draw_landmarks(
                image,
                results.pose_landmarks,
                _mp_pose().POSE_CONNECTIONS
            )

            # UI box
            cv2.rectangle(image, (0, 0), (260, 80), (0, 0, 0), -1)

            cv2.putText(image, 'HIGH KNEES', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(image, f'Reps: {counter}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('High Knees Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
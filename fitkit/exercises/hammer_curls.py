import cv2, mediapipe as mp, numpy as np


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


# ✅ STANDARD FUNCTION NAME
def hammer_curls_detection():
    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None

    with _mp_pose().Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[_mp_pose().PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[_mp_pose().PoseLandmark.RIGHT_SHOULDER.value].y]

                elbow = [landmarks[_mp_pose().PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[_mp_pose().PoseLandmark.RIGHT_ELBOW.value].y]

                wrist = [landmarks[_mp_pose().PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[_mp_pose().PoseLandmark.RIGHT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                if angle < 40 and stage == "down":
                    stage = "up"
                    counter += 1

            except:
                pass

            _mp_drawing().draw_landmarks(image, results.pose_landmarks, _mp_pose().POSE_CONNECTIONS)

            cv2.putText(image, f"Reps: {counter}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Hammer Curls", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
import cv2

MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Landmarks for shoulders and hips to define the torso
TORSO = [(11, 12), (11, 23), (12, 24), (23, 24)]


def _ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading pose model ({MODEL_URL}) ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Pose model downloaded.")


def _draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for a, b in TORSO:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (255, 140, 0), 2)
    for i in [11, 12, 23, 24]:
        if i in pts:
            cv2.circle(frame, pts[i], 5, (255, 200, 0), -1)


class BodyDetector:
    def __init__(self):
        _ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

    def detect(self, frame) -> bool:
        self._frame_ts += 33
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.pose_landmarks:
            return False

        _draw_pose(frame, result.pose_landmarks[0])
        return True

    def close(self):
        self._landmarker.close()

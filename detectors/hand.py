import math
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Landmarks (MediaPipe Hands)
TIP = [4, 8, 12, 16, 20]   # tips
PIP = [3, 6, 10, 14, 18]   # first joint (for detecting extended finger)
MCP = [2, 5, 9, 13, 17]    # base knuckles (for detecting clenched fist)

# Connections between landmarks to draw the hand skeleton
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading hand model ({MODEL_URL}) ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Hand model downloaded.")


def _draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


class HandDetector:
    def __init__(self, config: dict):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_hands=1,
            min_hand_detection_confidence=config.get("min_detection_confidence", 0.75),
            min_hand_presence_confidence=config.get("min_tracking_confidence", 0.6),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.6),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    @staticmethod
    def _dist(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _fingers_up(self, landmarks) -> list[bool]:
        wrist = landmarks[0]

        # Thumb: lateral tip and higher than the index finger's knuckle
        thumb_lateral = landmarks[TIP[0]].x < landmarks[MCP[0]].x
        thumb_high = landmarks[TIP[0]].y < landmarks[5].y
        thumb = thumb_lateral and thumb_high

        # Fingers 1-4: extended = tip is ≥1.6x further from the wrist than the base knuckle.
        # Using Euclidean distance to the wrist makes the detection orientation-independent
        # (works with thumb up, hand sideways, etc.)
        others = []
        for i in range(1, 5):
            tip_d = self._dist(landmarks[TIP[i]], wrist)
            mcp_d = self._dist(landmarks[MCP[i]], wrist)
            others.append(tip_d > mcp_d * 1.6)

        return [thumb] + others

    def _all_curled(self, landmarks) -> bool:
        wrist = landmarks[0]
        return all(
            self._dist(landmarks[TIP[i]], wrist) < self._dist(landmarks[MCP[i]], wrist) * 1.2
            for i in range(1, 5)
        )

    def _classify(self, fingers: list[bool], landmarks) -> str | None:
        thumb, idx, mid, ring, pinky = fingers

        # Fist: no fingers extended + tips clearly below the knuckles
        if not any(fingers) and self._all_curled(landmarks):
            return "fist"
        # Open hand: 4 main fingers extended (thumb optional)
        if idx and mid and ring and pinky:
            return "open_hand"
        # Thumb up: only thumb extended
        if thumb and not any((idx, mid, ring, pinky)):
            return "thumb_up"
        if not thumb and idx and not mid and not ring and not pinky:
            return "pointing"
        if not thumb and idx and mid and not ring and not pinky:
            return "peace"
        if thumb and idx and not mid and not ring and pinky:
            return "ok"
        return None

    def detect(self, frame) -> str | None:
        self._frame_ts += 33
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        _draw_hand(frame, landmarks)

        fingers = self._fingers_up(landmarks)
        return self._classify(fingers, landmarks)

    def close(self):
        self._landmarker.close()

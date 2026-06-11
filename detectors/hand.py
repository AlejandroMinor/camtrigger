import math
import urllib.request
from collections import deque
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

TIP = [4, 8, 12, 16, 20]
PIP = [3, 6, 10, 14, 18]
MCP = [2, 5, 9, 13, 17]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

HISTORY_SIZE = 6
STABLE_THRESHOLD = 4


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
        self._history: deque[str | None] = deque(maxlen=HISTORY_SIZE)
        self._zone_top: float = config.get("gesture_zone_top", 0.4)

    @staticmethod
    def _dist(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _fingers_up(self, landmarks) -> list[bool]:
        wrist = landmarks[0]

        # Thumb: compare tip vs IP joint distance to wrist — orientation-independent
        thumb_tip_d = self._dist(landmarks[TIP[0]], wrist)
        thumb_ip_d = self._dist(landmarks[PIP[0]], wrist)
        thumb = thumb_tip_d > thumb_ip_d * 1.3

        # Fingers 1-4: tip must be ≥1.5x further from wrist than the base knuckle
        others = []
        for i in range(1, 5):
            tip_d = self._dist(landmarks[TIP[i]], wrist)
            mcp_d = self._dist(landmarks[MCP[i]], wrist)
            others.append(tip_d > mcp_d * 1.5)

        return [thumb] + others

    def _all_curled(self, landmarks) -> bool:
        wrist = landmarks[0]
        return all(
            self._dist(landmarks[TIP[i]], wrist) < self._dist(landmarks[MCP[i]], wrist) * 1.2
            for i in range(1, 5)
        )

    def _classify(self, fingers: list[bool], landmarks) -> str | None:
        thumb, idx, mid, ring, pinky = fingers

        if not any(fingers) and self._all_curled(landmarks):
            return "fist"
        if idx and mid and ring and pinky:
            return "open_hand"
        if thumb and not any((idx, mid, ring, pinky)):
            return "thumb_up"
        if not thumb and idx and not mid and not ring and not pinky:
            return "pointing"
        if not thumb and idx and mid and not ring and not pinky:
            return "peace"
        if thumb and idx and not mid and not ring and pinky:
            return "ok"
        return None

    def _stable_gesture(self) -> str | None:
        counts: dict[str, int] = {}
        for g in self._history:
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        if not counts:
            return None
        best = max(counts, key=counts.get)
        return best if counts[best] >= STABLE_THRESHOLD else None

    def detect(self, frame) -> str | None:
        self._frame_ts += 33
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.hand_landmarks:
            self._history.clear()
            return None

        landmarks = result.hand_landmarks[0]

        if landmarks[0].y < self._zone_top:
            self._history.clear()
            return None

        _draw_hand(frame, landmarks)

        fingers = self._fingers_up(landmarks)
        gesture = self._classify(fingers, landmarks)
        self._history.append(gesture)
        return self._stable_gesture()

    def close(self):
        self._landmarker.close()

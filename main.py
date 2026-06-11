import cv2
import yaml
import time
import sys
import subprocess
from pathlib import Path

from detectors.hand import HandDetector
from detectors.body import BodyDetector
from actions.system import ActionRunner

CONFIG_PATH = Path(__file__).parent / "config.yaml"

GRACE_PERIOD = 0.15  # Seconds to keep showing a gesture after it disappears (to avoid flickering)


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _draw_hold_bar(frame, progress: float, gesture: str, fired: bool = False):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.7)
    bar_h = 24
    x0 = (w - bar_w) // 2
    y0 = h - 50

    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * min(progress, 1.0))
    color = (0, 255, 180) if fired else (0, 200, 255) if progress >= 1.0 else (0, 200, 0)
    if fill > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (180, 180, 180), 1)
    cv2.putText(frame, gesture, (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def _draw_presence(frame, present: bool, away_in: float | None):
    h, w = frame.shape[:2]
    if present:
        label = "present"
        color = (0, 200, 0)
    else:
        label = "absent"
        color = (0, 0, 220)
    cv2.putText(frame, label, (w - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Remaining time before executing the away action
    if away_in is not None and away_in > 0:
        countdown = f"away in {away_in:.0f}s"
        cv2.putText(frame, countdown, (w - 160, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 220), 1)


def main():
    config = load_config()

    cam_cfg = config.get("camera", {})
    cooldown: float = config.get("cooldown", 3.0)
    hold_duration: float = config.get("hold_duration", 0.8)
    debug: bool = config.get("debug", True)

    presence_cfg = config.get("presence", {})
    away_timeout: float = presence_cfg.get("away_timeout", 10.0)
    away_command: str = presence_cfg.get("away_command", "")
    present_command: str = presence_cfg.get("present_command", "")

    hand_detector = HandDetector(config.get("hand", {}))
    body_detector = BodyDetector()
    action_runner = ActionRunner(config.get("gestures", {}))

    cap = cv2.VideoCapture(cam_cfg.get("index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

    if not cap.isOpened():
        print("Error: could not open camera.")
        sys.exit(1)

    print("Started. Press Q to quit.")

    # Gesture state
    current_gesture: str | None = None
    gesture_start: float | None = None
    last_seen: float = 0.0
    last_trigger_time: float = 0.0
    just_fired: bool = False

    # Presence state
    person_present: bool = True
    last_present_time: float = time.monotonic()
    away_triggered: bool = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        now = time.monotonic()

        # --- Presence detection ---
        person_detected = body_detector.detect(frame)

        if person_detected:
            if not person_present and away_triggered:
                # Return from away state
                person_present = True
                away_triggered = False
                if present_command:
                    print(f"[present] -> {present_command}")
                    subprocess.Popen(present_command, shell=True)
            elif not person_present:
                person_present = True
            last_present_time = now
        else:
            if person_present and (now - last_present_time) > away_timeout:
                person_present = False
                if not away_triggered:
                    away_triggered = True
                    if away_command:
                        print(f"[away] -> {away_command}")
                        subprocess.Popen(away_command, shell=True)

        # --- Gesture detection (only if person is present) ---
        gesture = hand_detector.detect(frame) if person_present else None

        if gesture:
            if gesture != current_gesture:
                current_gesture = gesture
                gesture_start = now
                just_fired = False
            last_seen = now
        else:
            if current_gesture and (now - last_seen) > GRACE_PERIOD:
                current_gesture = None
                gesture_start = None
                just_fired = False

        ready_to_fire = (now - last_trigger_time) >= cooldown

        if current_gesture and gesture_start and not just_fired and ready_to_fire:
            held = now - gesture_start
            progress = held / hold_duration

            if held >= hold_duration:
                action_runner.run(current_gesture)
                last_trigger_time = now
                just_fired = True
                progress = 1.0

            if debug:
                _draw_hold_bar(frame, progress, current_gesture, fired=just_fired)

        # --- Debug overlay ---
        if debug:
            label = current_gesture or "---"
            color = (0, 255, 0) if current_gesture else (100, 100, 100)
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            away_in = None
            if not person_present or (not person_detected and person_present):
                elapsed = now - last_present_time
                remaining = away_timeout - elapsed
                away_in = remaining if remaining > 0 else None

            _draw_presence(frame, person_present or person_detected, away_in)
            cv2.imshow("Camera Control [Q para salir]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_detector.close()
    body_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

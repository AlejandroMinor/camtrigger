import subprocess


class ActionRunner:
    def __init__(self, config: dict):
        self._gestures_config = config

    def run(self, gesture: str) -> None:
        gesture_cfg = self._gestures_config.get(gesture)
        if not gesture_cfg:
            return

        command = gesture_cfg.get("command", "")
        description = gesture_cfg.get("description", gesture)

        if not command:
            return

        print(f"[{gesture}] {description} -> {command}")
        subprocess.Popen(command, shell=True)

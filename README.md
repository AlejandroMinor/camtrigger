# camtrigger

System control via hand gestures and presence detection using a webcam. Uses MediaPipe for hand and body recognition, executes configurable shell commands.

The default configuration uses desktop notifications to demonstrate each gesture. Replace any `command` with whatever you want — media control, window management, scripts, etc.

## Dependencies

```bash
sudo pacman -S python-opencv python-yaml
yay -S python-mediapipe-bin
```

MediaPipe models (~30 MB total) are downloaded automatically to `models/` on first run.

## Usage

```bash
python3 main.py
```

A window opens with the camera feed. The detected gesture appears top-left, presence status top-right. A progress bar fills as you hold a gesture — the command fires when it completes. Press **Q** to quit.

## Available gestures

| Gesture     | Description                      |
|-------------|----------------------------------|
| `thumb_up`  | Only thumb extended upward       |
| `fist`      | All fingers closed               |
| `open_hand` | All fingers open                 |
| `peace`     | Index and middle finger extended |
| `pointing`  | Only index finger extended       |

## Configuration

Edit `config.yaml`:

```yaml
cooldown: 3.0       # seconds to wait after triggering an action
hold_duration: 0.8  # seconds to hold a gesture before it triggers
debug: true

presence:
  away_timeout: 10                            # seconds without detection before away
  away_command: "loginctl lock-session"       # runs when you leave
  present_command: "notify-send 'Welcome back!'"  # runs when you return

gestures:
  open_hand:
    command: "playerctl play-pause"
    description: Pause music

  fist:
    command: "pactl set-sink-mute @DEFAULT_SINK@ toggle"
    description: Toggle mute
```

The `command` field accepts any shell command.

## Structure

```
camera/
├── main.py              # main loop
├── config.yaml          # configuration and gesture -> command mapping
├── models/              # auto-downloaded models
├── detectors/
│   ├── hand.py          # hand gesture detection (MediaPipe Hands)
│   └── body.py          # presence detection (MediaPipe Pose)
└── actions/
    └── system.py        # shell command execution
```

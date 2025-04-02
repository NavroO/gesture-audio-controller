# 🎶 Gesture-Controlled Music Player

Control your music **without touching anything** – just move your hands!  
This Python app lets you control the **volume** and **playback speed** of a song using **hand gestures** captured by your webcam.

---

## 🖐 Features

- ✋ **Volume Control** – Touch your **left thumb and index finger** to reduce volume, move them apart to increase it.
- 👐 **Playback Speed Control** – Move both hands closer together to slow down, move them apart to speed up.
- 🎵 Plays any local `.mp3` file using `pygame`.

---

## 🚀 How It Works

- Uses your webcam to detect hand gestures.
- Processes finger and hand positions using OpenCV (optionally MediaPipe for higher accuracy).
- Maps distances between fingers/hands to audio volume and speed.

---

## 📦 Requirements

- Python 3.8+
- `pygame`
- `opencv-python`
- (optional but recommended) `mediapipe` – for more accurate hand tracking

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage
```python app.py ./your-song.mp3```


## 📂 Project Structure
├── app.py                # Entry point
├── README.md
├── requirements.txt
└── assets/                # (Optional) Example MP3s

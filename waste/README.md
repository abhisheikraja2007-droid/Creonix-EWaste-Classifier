## E-Waste Real-Time Classification (YOLOv8-cls)

Classify e-waste items into classes (Battery, Keyboard, etc.) and convert to a high-level decision: Recycle vs Waste. Works with DroidCam (USB/IP) in real time.

### 1) Setup

- Python 3.10/3.11 recommended
- Create venv and install requirements:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Dataset Layout

Your data is already in the expected structure:

```
data/
  train/
    Battery/ ... jpg
    Keyboard/ ...
    ...
  val/
    Battery/ ...
    ...
  test/
    Battery/ ...
    ...
```

### 3) Train

```powershell
python train_cls.py --data data --model yolov8n-cls.pt --epochs 30 --imgsz 224 --batch 32
```

Weights saved at `runs/cls/ewaste/weights/best.pt`.

### 4) Real-Time Inference

- DroidCam USB: appears as a webcam device (index 0/1/2). Try `--source 0` or `--source 1`.
- DroidCam Wi‑Fi/IP: use the video URL like `http://192.168.0.50:4747/video`.

```powershell
python realtime_cls.py --weights runs/cls/ewaste/weights/best.pt --source 0 --imgsz 224 --decision-map decision_map.yaml
```

The overlay shows top class, confidence, and decision (Recycle/Waste). Edit `decision_map.yaml` to map classes to decisions.

### 5) Notes

- If you prefer detection with bounding boxes, we can switch to YOLOv8 detection (needs different training pipeline).
- If the camera fails to open on Windows, try different indices and check DroidCam drivers.


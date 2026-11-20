from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import base64
import time
from datetime import datetime
import json
import yaml
from ultralytics import YOLO
import threading
import queue
import numpy as np
import subprocess
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ewaste_classification_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
CONFIG = {
    'weights': 'runs/cls/ewaste/weights/best.pt',
    'source': '0',  # Laptop camera (usually index 0)
    'imgsz': 224,
    'decision_map': 'decision_map.yaml',
    'conf': 0.25,
    'min_conf': 0.60,
    'stable_frames': 3,
    'skip_face': True,
    'face_area_thresh': 0.04
}

# Global variables
model = None
cap = None
prediction_queue = queue.Queue()
is_streaming = False
stream_thread = None
realtime_process = None
cached_classes = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# Pre-load model to ensure classes are available
def init_model():
    global model, cached_classes
    try:
        print("Loading YOLO model...")
        model = YOLO(CONFIG['weights'])
        cached_classes = list(model.names.values())
        print(f"Model loaded successfully. Classes: {cached_classes}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

# Initialize model on startup (in background)
import threading
def load_model_background():
    init_model()

model_thread = threading.Thread(target=load_model_background, daemon=True)
model_thread.start()

def load_decision_map(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return {str(k): str(v).lower() for k, v in data.items()}
    except FileNotFoundError:
        return {}

def normalize_name(name: str) -> str:
    n = name.strip().lower()
    n = n.replace('-', ' ').replace('_', ' ')
    n = ' '.join(n.split())
    return n

def resolve_decision(class_name: str, decision_map, default_decision: str = 'recycle') -> str:
    candidates = [
        class_name,
        class_name.strip(),
        class_name.lower(),
        class_name.replace('_', ' '),
        class_name.replace('-', ' '),
        class_name.replace('_', ' ').lower(),
        class_name.replace('-', ' ').lower(),
        normalize_name(class_name),
    ]
    normalized_map = {normalize_name(k): v for k, v in decision_map.items()}
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized_map:
            v = normalized_map[key]
            return v if v in ('recycle', 'waste') else default_decision
    return default_decision

def load_face_cascade():
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        return face_cascade
    except Exception:
        return None

def detect_face(frame, face_cascade):
    if face_cascade is None:
        return False, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return False, 0.0
    h, w = frame.shape[:2]
    area_total = float(h * w)
    area_faces = 0.0
    for (x, y, fw, fh) in faces:
        area_faces += (fw * 2) * (fh * 2)
    return True, min(1.0, area_faces / max(1.0, area_total))

def get_trained_classes():
    """Get list of trained classes from the model (cached for speed)"""
    global cached_classes
    if cached_classes is not None:
        return cached_classes
    
    # Fallback if cache is empty
    return ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

def start_realtime_detection():
    """Start the realtime_cls.py script"""
    global realtime_process
    try:
        # Stop any existing process first
        if realtime_process and realtime_process.poll() is None:
            realtime_process.terminate()
            realtime_process = None
        
        # Find working camera index, prefer laptop cam at 0
        camera_index = find_working_camera()
        if camera_index is None:
            camera_index = 0  # Default fallback to laptop cam
        
        # Start realtime_cls.py with appropriate parameters
        cmd = [
            'python', 'realtime_cls.py',
            '--weights', CONFIG['weights'],
            '--source', str(camera_index),
            '--imgsz', str(CONFIG['imgsz']),
            '--decision-map', CONFIG['decision_map'],
            '--min-conf', str(CONFIG['min_conf']),
            '--stable-frames', str(CONFIG['stable_frames']),
            '--trigger', 'none'  # Always on detection
        ]
        
        # Add skip-face flag if enabled
        if CONFIG['skip_face']:
            cmd.append('--skip-face')
        
        print(f"🚀 Starting realtime detection with: {' '.join(cmd)}")
        
        # Start the process
        realtime_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Start thread to monitor output
        threading.Thread(target=monitor_realtime_output, daemon=True).start()
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if realtime_process.poll() is None:
            print("✅ Realtime detection started successfully")
            return True
        else:
            stdout, stderr = realtime_process.communicate()
            print(f"❌ Process failed to start.")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting realtime detection: {e}")
        return False

def stop_realtime_detection():
    """Stop the realtime_cls.py script"""
    global realtime_process
    try:
        if realtime_process and realtime_process.poll() is None:
            realtime_process.terminate()
            realtime_process.wait(timeout=5)
            print("Realtime detection stopped")
        realtime_process = None
        return True
    except Exception as e:
        print(f"Error stopping realtime detection: {e}")
        return False

def monitor_realtime_output():
    """Monitor output from realtime_cls.py and emit predictions"""
    global realtime_process
    if not realtime_process:
        return
    
    try:
        print("🔍 Monitoring realtime output...")
        for line in iter(realtime_process.stdout.readline, ''):
            if line.strip():
                print(f"📝 Realtime output: {line.strip()}")
                # Parse prediction output
                if line.startswith("PREDICTION:"):
                    try:
                        prediction_json = line.replace("PREDICTION:", "").strip()
                        prediction_data = json.loads(prediction_json)
                        
                        # Add color for frontend
                        prediction_data['color'] = 'success' if prediction_data['decision'] == 'recycle' else 'danger'
                        
                        # Emit to frontend
                        socketio.emit('prediction', prediction_data)
                        print(f"🎯 Emitted prediction: {prediction_data}")
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing prediction JSON: {e}")
                elif "error" in line.lower() or "failed" in line.lower():
                    print(f"⚠️ Realtime script error: {line.strip()}")
    except Exception as e:
        print(f"❌ Error monitoring realtime output: {e}")

def find_working_camera():
    """Try different camera indices to find a working one, prioritizing laptop camera"""
    # First try laptop camera (index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.release()
            print("Found laptop camera at index 0")
            return 0
        cap.release()
    
    # If laptop camera fails, try other indices
    for i in range(1, 5):  # Try indices 1-4
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                print(f"Found camera at index {i}")
                return i
            cap.release()
    return None

def video_stream():
    global cap, model, is_streaming
    
    if model is None:
        model = YOLO(CONFIG['weights'])
    
    decision_map = load_decision_map(CONFIG['decision_map'])
    face_cascade = load_face_cascade() if CONFIG['skip_face'] else None
    
    # Try to find a working camera
    camera_index = find_working_camera()
    if camera_index is None:
        print("No working camera found. Please check camera permissions and connection...")
        # Create a placeholder frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Found", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Check laptop camera", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(placeholder, "permissions & connection", (140, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(placeholder, "Refresh page to retry", (160, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        while is_streaming:
            _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    print(f"Using camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return
    
    # Optimize camera settings for better FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    is_streaming = True
    last_class_name = None
    stable_count = 0
    frame_skip = 2  # Process every 2nd frame for better FPS
    
    frame_count = 0
    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for better FPS (only process every nth frame)
        if frame_count % frame_skip != 0:
            # Still send frame to maintain video feed
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue
        
        # Face detection skip (only on processed frames)
        if CONFIG['skip_face'] and face_cascade:
            has_face, face_ratio = detect_face(frame, face_cascade)
            if has_face and face_ratio >= CONFIG['face_area_thresh']:

                continue
        
        # Inference (only on processed frames)
        results = model(frame, imgsz=CONFIG['imgsz'], verbose=False)
        result = results[0]
        
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            best_idx = int(probs.argmax())
            best_conf = float(probs[best_idx])
            class_name = result.names[best_idx]
            
            if best_conf >= max(CONFIG['conf'], CONFIG['min_conf']):
                if last_class_name == class_name:
                    stable_count += 1
                else:
                    last_class_name = class_name
                    stable_count = 1
                
                if stable_count >= CONFIG['stable_frames']:
                    decision = resolve_decision(class_name, decision_map)
                    confidence = best_conf * 100
                    
                    # Emit prediction to frontend
                    prediction_data = {
                        'class_name': class_name,
                        'confidence': round(confidence, 1),
                        'decision': decision,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'color': 'success' if decision == 'recycle' else 'danger'
                    }
                    
                    socketio.emit('prediction', prediction_data)
                    stable_count = 0  # Reset after emitting
            else:
                stable_count = max(0, stable_count - 1)
        
        # Encode frame for streaming (lower quality for better FPS)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    # Return a placeholder since we're using realtime_cls.py for detection
    def placeholder_stream():
        while True:
            # Create a simple placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera feed handled by realtime_cls.py", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(placeholder, "Click 'Start Detection' to begin", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    
    return Response(placeholder_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/classes')
def get_classes():
    classes = get_trained_classes()
    return {'classes': classes}

@app.route('/api/cameras')
def get_cameras():
    """Get list of available cameras"""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cameras.append({
                    'index': i,
                    'name': f'Camera {i}',
                    'working': True
                })
            cap.release()
        else:
            cameras.append({
                'index': i,
                'name': f'Camera {i}',
                'working': False
            })
    return {'cameras': cameras}

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to E-Waste Classification System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_stream')
def handle_start_stream():
    global is_streaming
    print("Starting realtime detection...")
    if start_realtime_detection():
        is_streaming = True
        emit('status', {'message': 'Realtime detection started successfully!', 'type': 'success'})
        print("✅ Realtime detection started")
    else:
        emit('status', {'message': 'Failed to start detection. Check console for errors.', 'type': 'error'})
        print("❌ Failed to start realtime detection")

@socketio.on('stop_stream')
def handle_stop_stream():
    global is_streaming
    print("Stopping realtime detection...")
    if stop_realtime_detection():
        is_streaming = False
        emit('status', {'message': 'Realtime detection stopped', 'type': 'info'})
        print("✅ Realtime detection stopped")
    else:
        emit('status', {'message': 'Failed to stop detection', 'type': 'error'})
        print("❌ Failed to stop realtime detection")

if __name__ == '__main__':
    print("Starting E-Waste Classification Dashboard...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, debug=False, host='127.0.0.1', port=5000)



import argparse
import time
from typing import Dict, Optional, Tuple
import cv2
import yaml
import json
from ultralytics import YOLO


def load_decision_map(path: str) -> Dict[str, str]:
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


def resolve_decision(class_name: str, decision_map: Dict[str, str], default_decision: str) -> str:
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


def get_capture(source: str):
    # Try requested source first
    if source.isdigit():
        requested_index = int(source)
        cap = cv2.VideoCapture(requested_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        # Fallback: try common indices (0-4), prioritizing 0
        try_indices = [0, 1, 2, 3, 4]
        if requested_index in try_indices:
            try_indices.remove(requested_index)
        for idx in try_indices:
            cap_fallback = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap_fallback.isOpened():
                return cap_fallback
        return cap  # return last attempted (closed) cap for caller to handle
    else:
        cap = cv2.VideoCapture(source)
        return cap


def list_cameras(max_index: int = 10):
    print('Scanning camera indices (DirectShow backend)...')
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        ok, frame = cap.read() if opened else (False, None)
        shape = frame.shape if ok else None
        print(f'index {idx}: opened={opened}, frame_ok={ok}, shape={shape}')
        if opened:
            cap.release()


def load_face_cascade() -> Optional[cv2.CascadeClassifier]:
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        return face_cascade
    except Exception:
        return None


def detect_face(frame, face_cascade: Optional[cv2.CascadeClassifier]) -> Tuple[bool, float]:
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


def main():
    parser = argparse.ArgumentParser(description='Real-time e-waste classification (Recycle vs Waste)')
    parser.add_argument('--weights', type=str, default='runs/cls/ewaste/weights/best.pt', help='Path to trained weights')
    parser.add_argument('--source', type=str, default='0', help='Camera index (e.g., 0/1/2) or IP stream URL')
    parser.add_argument('--imgsz', type=int, default=224, help='Inference image size')
    parser.add_argument('--decision-map', type=str, default='decision_map.yaml', help='YAML mapping class->decision')
    parser.add_argument('--conf', type=float, default=0.25, help='Minimum confidence to show label')
    parser.add_argument('--list-cams', action='store_true', help='List available camera indices and exit')
    parser.add_argument('--min-conf', type=float, default=0.60, help='Gate: minimum confidence to consider a prediction')
    parser.add_argument('--stable-frames', type=int, default=3, help='Gate: require same class for N consecutive frames')
    parser.add_argument('--skip-face', action='store_true', help='Skip predictions when a face is detected')
    parser.add_argument('--face-area-thresh', type=float, default=0.04, help='Skip if face area exceeds this fraction of frame')
    parser.add_argument('--default-decision', type=str, default='recycle', choices=['recycle', 'waste'], help='Fallback decision when class not in map')
    parser.add_argument('--trigger', type=str, default='key', choices=['none','key'], help="When to run inference: 'none' = always, 'key' = only while key held")
    parser.add_argument('--trigger-key', type=str, default='o', help="Key to hold for inference when --trigger key")
    args = parser.parse_args()

    if args.list_cams:
        list_cameras(10)
        return

    decision_map = load_decision_map(args.decision_map)

    model = YOLO(args.weights)

    cap = get_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video source: {args.source}')

    face_cascade = load_face_cascade() if args.skip_face else None

    fps_time = time.time()
    frame_count = 0

    last_class_name: Optional[str] = None
    stable_count: int = 0
    current_label_text = 'Hold O to classify an e-waste item' if args.trigger == 'key' else 'Show an e-waste item to classify'
    current_color = (255, 255, 255)

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Stream ended or cannot read frame.')
            break

        # Read key early for trigger handling
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # Trigger gating: only infer if trigger condition satisfied
        trigger_ok = True
        if args.trigger == 'key':
            trigger_ok = (key == ord(args.trigger_key.lower())) or (key == ord(args.trigger_key.upper()))

        if not trigger_ok:
            # Show waiting banner and continue without inference
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w - 10, 80), (0, 0, 0), -1)
            wait_text = f"Hold '{args.trigger_key.upper()}' to classify"
            cv2.putText(frame, wait_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow('E-Waste Recycle vs Waste - YOLOv8-cls', frame)
            continue

        if args.skip_face:
            has_face, face_ratio = detect_face(frame, face_cascade)
            if has_face and face_ratio >= args.face_area_thresh:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10, 10), (w - 10, 80), (0, 0, 0), -1)
                cv2.putText(frame, 'Waiting for object (face detected)', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow('E-Waste Recycle vs Waste - YOLOv8-cls', frame)
                continue

        results = model(frame, imgsz=args.imgsz, verbose=False)
        result = results[0]

        label_text = None
        color = (255, 255, 255)

        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            best_idx = int(probs.argmax())
            best_conf = float(probs[best_idx])
            class_name = result.names[best_idx]

            if best_conf >= max(args.conf, args.min_conf):
                if last_class_name == class_name:
                    stable_count += 1
                else:
                    last_class_name = class_name
                    stable_count = 1

                if stable_count >= args.stable_frames:
                    decision = resolve_decision(class_name, decision_map, args.default_decision)
                    label_text = f"{class_name}: {best_conf*100:.1f}% -> {decision.capitalize()}"
                    color = (0, 200, 0) if decision == 'recycle' else (0, 0, 255)
                    current_label_text = label_text
                    current_color = color
                    
                    # Output prediction for webapp to parse
                    prediction_data = {
                        'class_name': class_name,
                        'confidence': round(best_conf * 100, 1),
                        'decision': decision,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    print(f"PREDICTION:{json.dumps(prediction_data)}")
            else:
                stable_count = max(0, stable_count - 1)
        else:
            stable_count = 0

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.putText(frame, current_label_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2, cv2.LINE_AA)

        frame_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            fps_time = now
            frame_count = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow('E-Waste Recycle vs Waste - YOLOv8-cls', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

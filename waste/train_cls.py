import argparse
from ultralytics import YOLO

try:
    import torch
except Exception:  # torch is an optional explicit import here; ultralytics depends on it
    torch = None


def resolve_device(user_device: str) -> str:
    # Accept explicit 'cpu' or CUDA index like '0'
    if user_device and user_device.lower() != 'auto':
        return user_device
    # Auto: choose CUDA:0 if available, else CPU
    if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
        return '0'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 classification on e-waste dataset")
    parser.add_argument('--data', type=str, default='data', help='Path to dataset root containing train/val/test')
    parser.add_argument('--model', type=str, default='yolov8n-cls.pt', help='Base classification model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size for training')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--project', type=str, default='runs/cls', help='Project directory for runs')
    parser.add_argument('--name', type=str, default='ewaste', help='Run name')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', or CUDA device index like '0'")
    args = parser.parse_args()

    device = resolve_device(args.device)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=device
    )

    print("Training complete. Check runs in:", f"{args.project}/{args.name}")


if __name__ == '__main__':
    main()

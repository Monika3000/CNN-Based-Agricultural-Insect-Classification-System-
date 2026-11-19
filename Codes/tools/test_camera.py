"""Capture frames from the webcam and run the detector on them (no GUI needed)."""
import sys, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_detect import PestDetector

out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "cam_frames"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("camera 0 not available"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
det = PestDetector(conf=0.75, margin=0.15)
print("capturing 5 frames...")
for i in range(5):
    ok, frame = cap.read()
    if not ok:
        print("frame read failed"); break
    p = out_dir / f"cam_{i}.jpg"
    cv2.imwrite(str(p), frame)
    dets = det.detect(frame)
    print(f"frame {i}: {frame.shape} -> {[(d['label'], round(d['conf'],3), d['box']) for d in dets]}")
    time.sleep(0.3)
cap.release()
print("saved frames to", out_dir)

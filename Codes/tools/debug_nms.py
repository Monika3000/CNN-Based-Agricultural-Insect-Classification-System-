import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_detect import PestDetector

det = PestDetector(conf=0.75, margin=0.15)
orig = det._classify
calls = []
def traced(batch):
    out = orig(batch)
    for c, o in zip(batch, out):
        calls.append((c.shape[:2], det.classes[int(o.argmax())], float(o.max())))
    return out
det._classify = traced

frame = cv2.imread(str(Path(__file__).resolve().parent.parent / "artifacts/test_scenes/scene_aphids.jpg"))
dets = det.detect(frame, conf=0.75)
print("classify calls:")
for c in calls:
    print("   crop", c[0], "->", c[1], round(c[2], 3))
print("final:", [(d["label"], round(d["conf"], 3), d["box"]) for d in dets])

"""
Capture-on-demand webcam pest classifier for CNN_OP.
- Shows live preview until 'c' is pressed
- On 'c': capture exactly one frame, classify with existing EfficientNet-B1
- Display result, return to preview
- 'q' or ESC to quit
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent
ART = BASE / "artifacts"
DB = json.loads((BASE / "pesticide_db.json").read_text())

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Classifier:
    def __init__(self, ckpt_path=None, device=None):
        self.device = device or torch.device("cpu")
        ckpt_path = Path(ckpt_path) if ckpt_path else (ART / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.img_size = ckpt.get("img_size", 240)
        self.model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        print(f"[classifier] {ckpt_path.name} | {ckpt['model']} | {len(self.classes)} classes | {self.classes}")

    @torch.no_grad()
    def predict(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x = self.tf(img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        vals, idx = probs.topk(3)
        results = []
        for v, i in zip(vals, idx):
            results.append({"class": self.classes[i], "conf": float(v)})
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    clf = Classifier(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press C to capture and classify | Press Q to quit")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped")
            break
        disp = frame.copy()
        cv2.putText(disp, "Press C to capture and classify | Press Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Light Trap - Pest Detection", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):  # q or ESC
            break
        elif k == ord('c'):
            # Capture exactly one frame and classify
            results = clf.predict(frame)
            best = results[0]
            # Show result
            res = frame.copy()
            h, w = res.shape[:2]
            cv2.rectangle(res, (0, 0), (w, 80), (0, 0, 0), -1)
            text = f"{best['class']}  {best['conf']*100:.1f}%"
            cv2.putText(res, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(res, "Press C for another | Press Q to quit", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Light Trap - Pest Detection", res)
            # Wait for next key
            while True:
                kk = cv2.waitKey(100) & 0xFF
                if kk in (ord('q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif kk == ord('c'):
                    break
            # Continue to preview
            continue
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

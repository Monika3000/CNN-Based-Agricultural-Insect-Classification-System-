"""
Build a synthetic 'printout in front of the webcam' test scene:
  - white A4-like paper filling part of the frame
  - a pest image printed on it (from the TRAIN folder only)
  - realistic camera artefacts: slight rotation, uneven lighting, blur, noise
Used to verify live_detect.py end-to-end without a camera.
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np

TRAIN = Path(r"C:\Users\Monika\Desktop\Agriculture Project\BAU-Insectv2\extracted_folder\pest\train")
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "test_scenes"
OUT.mkdir(parents=True, exist_ok=True)


def make_scene(cls, img_name, out_name, frame_w=1280, frame_h=720, paper_frac=0.62,
               pos=(0.55, 0.5), rot=4.0, seed=0):
    rng = np.random.default_rng(seed)
    files = sorted((TRAIN / cls).glob("*.jpg"))
    src = TRAIN / cls / img_name if img_name else files[0]
    insect = cv2.imread(str(src))
    if insect is None:
        raise SystemExit(f"cannot read {src}")

    # paper
    paper_w = int(frame_w * paper_frac)
    paper_h = int(paper_h_ratio := frame_h * 0.95)
    paper = np.full((paper_h, paper_w, 3), 248, np.uint8)

    # print the insect image on the paper (keep aspect)
    ih, iw = insect.shape[:2]
    target_h = int(paper_h * 0.62)
    scale = target_h / ih
    insect_r = cv2.resize(insect, (int(iw * scale), target_h), interpolation=cv2.INTER_CUBIC)
    px = (paper_w - insect_r.shape[1]) // 2
    py = int(paper_h * 0.16)
    paper[py:py + insect_r.shape[1] * 0 + insect_r.shape[0], px:px + insect_r.shape[1]] = insect_r

    # rotate the paper slightly
    M = cv2.getRotationMatrix2D((paper_w / 2, paper_h / 2), rot, 1.0)
    paper = cv2.warpAffine(paper, M, (paper_w, paper_h), borderValue=(248, 248, 248))

    # background (desk)
    bg = np.full((frame_h, frame_w, 3), 60, np.uint8)
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    x = int(frame_w * pos[0] - paper_w / 2)
    y = int(frame_h * pos[1] - paper_h / 2)
    x = int(np.clip(x, -paper_w + 200, frame_w - 200)); y = int(np.clip(y, -paper_h + 150, frame_h - 150))
    bg[y:y + paper_h, x:x + paper_w] = paper

    # camera artefacts: uneven lighting, slight blur, sensor noise
    yy, xx = np.mgrid[0:frame_h, 0:frame_w]
    grad = 1.25 - 0.45 * (xx / frame_w) - 0.25 * (yy / frame_h)
    bg = np.clip(bg.astype(np.float32) * grad[..., None], 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    noise = rng.normal(0, 4.5, bg.shape)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out = OUT / out_name
    cv2.imwrite(str(out), bg)
    # ground-truth box of the printed insect (after rotation approx -> use paper region)
    print(f"{out}  (printed {cls} from {src.name})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="stem_borer")
    ap.add_argument("--img", default=None)
    ap.add_argument("--out", default="scene.jpg")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_scene(a.cls, a.img, a.out, seed=a.seed)

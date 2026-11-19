"""Sanity-check the CLIP prompts against REAL training images (train folder only)."""
import json, random
from pathlib import Path
from PIL import Image
from clip_check import classify_zero_shot, insect_score

TRAIN = Path(r"C:\Users\Monika\Desktop\Agriculture Project\BAU-Insectv2\extracted_folder\pest\train")
N = 10
random.seed(0)
ok = 0; tot = 0; rows = []
for cls_dir in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
    files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    picks = random.sample(files, min(N, len(files)))
    hits = 0
    for f in picks:
        im = Image.open(f)
        s = classify_zero_shot(im)
        best = max(s, key=s.get)
        hits += (best == cls_dir.name)
        tot += 1; ok += (best == cls_dir.name)
        if f is picks[0]:
            top = sorted(s.items(), key=lambda x: -x[1])[:3]
            rows.append((cls_dir.name, [(k, round(v, 4)) for k, v in top]))
    print(f"{cls_dir.name:<12} zero-shot top1 = {hits}/{len(picks)}")
print(f"\nOverall CLIP zero-shot accuracy on real photos: {ok}/{tot} = {ok/tot:.3f}\n")
print("Example top-3 per class:")
for r in rows:
    print(" ", r[0], r[1])

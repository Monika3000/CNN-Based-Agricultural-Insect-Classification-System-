"""
Validate the downloaded pool with CLIP and promote the good files into the
training layout:

    data/_pool/cartoon/<class>/*.jpg  ->  data/cartoon/<class>/*.jpg   (if CLIP agrees)
    data/_pool/negatives/*.jpg         ->  data/negatives/*.jpg          (if CLIP says "not a pest")

Writes data/manifest_pool.json with one record per file (decision + CLIP scores).
"""
import json, shutil, sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_check import CLASS_PROMPTS, NEGATIVE_PROMPTS, clip_scores

BASE = Path(__file__).resolve().parent.parent
POOL = BASE / "data" / "_pool"
OUT_CARTOON = BASE / "data" / "cartoon"
OUT_NEG = BASE / "data" / "negatives"
MANIFEST = BASE / "data" / "manifest_pool.json"
DEV = "cpu"

CLASSES = list(CLASS_PROMPTS.keys())


def load(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    OUT_CARTOON.mkdir(parents=True, exist_ok=True)
    OUT_NEG.mkdir(parents=True, exist_ok=True)
    records = []

    # ---------- cartoons ----------
    for cdir in sorted((POOL / "cartoon").glob("*")):
        if not cdir.is_dir():
            continue
        cls = cdir.name
        files = sorted(cdir.glob("*.jpg"))
        if not files:
            print(f"{cls:12} 0 files")
            continue
        imgs, keep = [], []
        for f in files:
            im = load(f)
            if im is None:
                records.append({"file": str(f.relative_to(BASE)), "class": cls, "decision": "unreadable"})
                continue
            imgs.append(im); keep.append(f)
        if not imgs:
            continue
        sc = clip_scores(imgs, [CLASS_PROMPTS[c] for c in CLASSES], DEV)
        acc = 0
        for f, row in zip(keep, sc):
            order = np_argsort_desc(row)
            top, second = int(order[0]), int(order[1])
            margin = float(row[top] - row[second])
            ok = CLASSES[top] == cls and margin > 0.004
            rec = {"file": str(f.relative_to(BASE)), "class": cls,
                   "clip_top": CLASSES[top], "clip_margin": round(margin, 5),
                   "clip_scores": {CLASSES[i]: round(float(row[i]), 4) for i in range(len(CLASSES))},
                   "decision": "accept" if ok else "reject"}
            records.append(rec)
            if ok:
                dst = OUT_CARTOON / cls / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                acc += 1
        print(f"{cls:12} {acc:3}/{len(keep)} accepted")

    # ---------- negatives ----------
    ndir = POOL / "negatives"
    nfiles = sorted(ndir.glob("*.jpg"))
    imgs, keep = [], []
    for f in nfiles:
        im = load(f)
        if im is None:
            records.append({"file": str(f.relative_to(BASE)), "class": "negatives", "decision": "unreadable"})
            continue
        imgs.append(im); keep.append(f)
    if imgs:
        s_ins = clip_scores(imgs, [CLASS_PROMPTS[c] for c in CLASSES], DEV)
        s_neg = clip_scores(imgs, NEGATIVE_PROMPTS, DEV)
        acc = 0
        for f, ri, rn in zip(keep, s_ins, s_neg):
            i_max = float(ri.max()); n_max = float(rn.max())
            ok = n_max >= i_max           # looks more like a non-insect than any pest
            records.append({"file": str(f.relative_to(BASE)), "class": "negatives",
                            "clip_insect_max": round(i_max, 4), "clip_neg_max": round(n_max, 4),
                            "decision": "accept" if ok else "reject"})
            if ok:
                shutil.copy2(f, OUT_NEG / f.name)
                acc += 1
        print(f"{'negatives':12} {acc:3}/{len(keep)} accepted")

    MANIFEST.write_text(json.dumps(records, indent=1))
    print("manifest ->", MANIFEST, f"({len(records)} records)")


def np_argsort_desc(row):
    return sorted(range(len(row)), key=lambda i: -row[i])


if __name__ == "__main__":
    main()

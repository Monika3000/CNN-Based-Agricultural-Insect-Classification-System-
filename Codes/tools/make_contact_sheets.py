"""Build per-class contact sheets from the TRAIN folder ONLY (never reads pest/test)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

TRAIN = Path(r"C:\Users\Monika\Desktop\Agriculture Project\BAU-Insectv2\extracted_folder\pest\train")
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "class_sheets"
OUT.mkdir(parents=True, exist_ok=True)

COLS, ROWS, CELL = 6, 3, 180

for cls_dir in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
    files = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not files:
        continue
    picks = files[: COLS * ROWS]
    sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(picks):
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        im.thumbnail((CELL - 4, CELL - 4))
        x = (i % COLS) * CELL + 2
        y = (i // COLS) * CELL + 2
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + 2), f.name[:14], fill="red")
    out = OUT / f"{cls_dir.name}.jpg"
    sheet.save(out, quality=88)
    print(out, len(files), "images")

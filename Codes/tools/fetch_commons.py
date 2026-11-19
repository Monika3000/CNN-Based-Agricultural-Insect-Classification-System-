"""
Stage 1 of dataset build: fetch candidate CARTOON / ILLUSTRATED insect images and
NEGATIVE (non-insect) images from Wikimedia Commons (freely licensed).

  -> downloads into  data/_pool/cartoon/<class>/  and  data/_pool/negatives/
  -> writes       data/manifest_pool.json  (source page, licence, author, query)

Relevance is NOT judged here. Stage 2 (validate_images.py) applies keyword scoring,
CLIP zero-shot verification and pHash de-duplication before promoting files to
data/cartoon/<class>/ and data/negatives/.
"""
import json, time, re, sys
from pathlib import Path
from urllib.parse import quote
import requests
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
POOL = BASE / "data" / "_pool"
POOL.mkdir(parents=True, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
UA = "PestClassifierDatasetBuilder/1.0 (educational research; local student project)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

ALLOWED_LICENSE = re.compile(r"(public domain|cc0|cc[ -]?by([ -]?(sa))?|attribution)", re.I)

CLASS_QUERIES = {
    "aphids": ["aphid illustration", "aphids drawing", "aphid diagram", "aphid clipart",
               "Aphididae illustration", "aphid engraving", "greenfly drawing", "Aphis drawing"],
    "armyworm": ["armyworm caterpillar drawing", "armyworm illustration", "Spodoptera drawing",
                  "Mythimna drawing", "caterpillar larva illustration", "lepidoptera larva plate",
                  "armyworm moth drawing", "cutworm drawing"],
    "beetle": ["beetle illustration drawing", "Coleoptera illustration", "beetle clipart",
               "weevil drawing", "scarab beetle drawing", "leaf beetle drawing", "ladybird beetle drawing",
               "beetle anatomy diagram", "Coleoptera plate engraving"],
    "bollworm": ["Helicoverpa armigera drawing", "bollworm illustration", "Heliothis drawing",
                  "cotton bollworm moth", "noctuid caterpillar drawing", "cotton bollworm larva",
                  "American bollworm drawing", "Helicoverpa plate"],
    "grasshopper": ["grasshopper illustration drawing", "Acrididae illustration", "grasshopper clipart",
                    "locust drawing", "grasshopper engraving", "short-horned grasshopper illustration",
                    "Orthoptera plate engraving", "grasshopper anatomy"],
    "mites": ["spider mite drawing", "Tetranychus illustration", "red spider mite diagram",
              "Tetranychidae illustration", "mite anatomy drawing", "spider mite clipart",
              "Acari engraving", "two-spotted spider mite"],
    "mosquito": ["mosquito illustration drawing", "mosquito diagram", "Culex drawing",
                 "mosquito clipart", "Aedes mosquito drawing", "Anopheles drawing",
                 "Culicidae plate engraving", "mosquito anatomy diagram"],
    "sawfly": ["sawfly drawing", "Athalia rosae drawing", "turnip sawfly", "sawfly larva",
               "Tenthredinidae illustration", "sawfly plate engraving", "sawfly illustration",
               "Athalia lugens"],
    "stem_borer": ["Chilo partellus drawing", "rice stem borer", "stem borer caterpillar",
                   "Scirpophaga incertulas drawing", "maize stem borer illustration",
                   "Chilo suppressalis", "Sesamia inferens", "Ostrinia nubilalis drawing",
                   "Pyralidae plate engraving", "corn borer larva illustration"],
}

NEGATIVE_QUERIES = [
    "apple fruit isolated", "wooden chair isolated", "leather shoe photograph",
    "coffee cup photograph", "laptop computer photograph", "brick wall texture",
    "mountain landscape photograph", "vintage car photograph", "portrait face photograph",
    "stack of books photograph", "flowers in vase", "cutlery fork knife",
    "cows in field farm", "sunset sky clouds", "toolbox hammer wrench",
    "plastic bottle container", "carrot vegetable", "strawberry fruit bowl",
    "tractor machine farm", "clay pot with plant", "textile fabric roll",
    "glass water bottle", "metal bucket", "ball of yarn", "paper notebook pen",
    "soccer ball on grass", "stone rocks pile", "clock on wall", "traffic signal light",
]

MAX_PER_CLASS = 90


def search(query, limit=30):
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": query, "gsrnamespace": "6", "gsrlimit": str(limit),
              "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "768"}
    delay = 3.0
    for _ in range(5):
        try:
            r = SESSION.get(API, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(delay); delay *= 2; continue
            r.raise_for_status()
            return list(((r.json().get("query") or {}).get("pages") or {}).values())
        except requests.RequestException:
            time.sleep(delay); delay *= 2
    return []


def licence_ok(meta):
    lic = (meta.get("LicenseShortName", {}) or {}).get("value", "") or ""
    return bool(ALLOWED_LICENSE.search(lic)), lic


def download(url, dest):
    """Download with retry/backoff. Uses Wikimedia thumbnail service when possible."""
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=90)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1)); continue
            r.raise_for_status()
            dest.write_bytes(r.content)
            return len(r.content)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


def collect(group, queries, out_dir, cap):
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, records = set(), []
    for q in queries:
        if len(records) >= cap:
            break
        for p in search(q):
            if len(records) >= cap:
                break
            title = p.get("title", "")
            if title in seen:
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata") or {}
            mime, w, h = ii.get("mime", ""), ii.get("width", 0), ii.get("height", 0)
            if mime not in ("image/jpeg", "image/png", "image/svg+xml", "image/webp", "image/gif"):
                continue
            if mime != "image/svg+xml" and (w < 200 or h < 200):
                continue
            ok, lic = licence_ok(meta)
            if not ok:
                continue
            ext = ".png" if mime in ("image/svg+xml",) else (".webp" if mime == "image/webp" else ".jpg")
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.replace("File:", ""))[:70]
            dest = out_dir / f"{safe}{ext}"
            seen.add(title)
            if not dest.exists():
                url = f"{FILEPATH}{quote(title.replace('File:', ''))}?width=768"
                n = download(url, dest)
                if n is None:
                    print(f"  ! dl fail {title[:60]}")
                    continue
                time.sleep(1.0)
            try:
                with Image.open(dest) as im:
                    im.verify()
                with Image.open(dest) as im:
                    dims = im.size
                    im = im.convert("RGB")
                    if min(dims) < 200:
                        dest.unlink(missing_ok=True); continue
            except Exception:
                dest.unlink(missing_ok=True); continue
            author = re.sub(r"<[^>]+>", "", (meta.get("Artist", {}) or {}).get("value", "") or "")[:120]
            records.append({
                "group": group, "file": dest.name, "title": title, "query": q,
                "source_page": ii.get("descriptionurl"), "file_url": ii.get("url"),
                "download_url": f"{FILEPATH}{quote(title.replace('File:', ''))}?width=768",
                "license": lic, "author": author,
                "width": dims[0], "height": dims[1], "bytes": dest.stat().st_size,
            })
            print(f"  + {group:<10} {title[:58]:<58} {lic[:20]}")
        time.sleep(1.5)
    return records


def main():
    allrec = []
    for cls, queries in CLASS_QUERIES.items():
        print(f"[cartoon] {cls}")
        allrec += collect(cls, queries, POOL / "cartoon" / cls, MAX_PER_CLASS)
    print("[negative] negatives")
    allrec += collect("negatives", NEGATIVE_QUERIES, POOL / "negatives", 220)

    manifest = BASE / "data" / "manifest_pool.json"
    manifest.write_text(json.dumps(allrec, indent=2))
    from collections import Counter
    print("\nPooled:", dict(Counter(r["group"] for r in allrec)))
    print("Manifest ->", manifest)


if __name__ == "__main__":
    main()

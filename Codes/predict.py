"""Inference: classify a pest image -> pesticide recommendation from pesticide_db.json."""
import json, sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import timm

BASE = Path(__file__).parent
ART = BASE / "artifacts"
DB = json.loads((BASE / "pesticide_db.json").read_text())

def load_model(device):
    ckpt = torch.load(ART / "best_model.pt", map_location=device, weights_only=False)
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt["img_size"]

@torch.no_grad()
def predict(image_path, topk=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, img_size = load_model(device)
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0]
    vals, idx = probs.topk(topk)
    results = [{"class": classes[i], "confidence": float(v)} for v, i in zip(vals, idx)]
    best = results[0]["class"]
    return best, results, DB.get(best, {})

def show(image_path):
    best, results, info = predict(image_path)
    print(f"\nImage: {image_path}")
    print("Top predictions:")
    for r in results:
        print(f"  {r['class']:<12} {r['confidence']*100:5.1f}%")
    print(f"\nPEST: {info.get('common_name', best)}")
    print(f"Damage: {info.get('damage','')}")
    print("Cultural/IPM:", "; ".join(info.get("ipm_cultural", [])))
    print("Biological:", "; ".join(info.get("biological", [])))
    print("Chemical options:")
    for c in info.get("chemical", []):
        print(f"  - {c['product']}: {c['dose']} ({c.get('note','')})")
    print("\nNOTE: Always confirm with the product label + local agri officer; respect PHI & PPE.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image2 ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        show(p)

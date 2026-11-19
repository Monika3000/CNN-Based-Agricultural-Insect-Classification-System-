"""Farmer-friendly dashboard: upload pest image -> insect + pesticide recommendation."""
import base64
import io
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import timm
from flask import Flask, request, jsonify

BASE = Path(__file__).parent
ART = BASE / "artifacts"
DB = json.loads((BASE / "pesticide_db.json").read_text())
DISCLAIMER = DB.pop("_disclaimer", "")
DB.pop("_sources", None)

app = Flask(__name__)
_model = None
_classes = None
_img_size = 240
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    global _model, _classes, _img_size
    if _model is None:
        ckpt = torch.load(ART / "best_model.pt", map_location=_device, weights_only=False)
        _model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["classes"]))
        _model.load_state_dict(ckpt["state_dict"])
        _model.to(_device).eval()
        _classes = ckpt["classes"]
        _img_size = ckpt.get("img_size", 240)
        print(f"Loaded {ckpt['model']} on {_device}, classes={_classes}")
    return _model, _classes, _img_size

def predict_pil(img: Image.Image, topk=3):
    model, classes, img_size = get_model()
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    vals, idx = probs.topk(topk)
    results = [{"class": classes[i], "confidence": round(float(v) * 100, 1)} for v, i in zip(vals, idx)]
    best = results[0]["class"]
    return best, results, DB.get(best, {})

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Light Trap - Pest Detection</title>
<style>
*{box-sizing:border-box}body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f1f8e9;color:#212121}
header{background:#2e7d32;color:#fff;padding:18px 24px}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.9;font-size:14px}
main{max-width:900px;margin:24px auto;padding:0 16px;display:grid;gap:16px;grid-template-columns:1fr 1fr}
@media(max-width:760px){main{grid-template-columns:1fr}}
.card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
#drop{border:2px dashed #66bb6a;border-radius:12px;padding:28px;text-align:center;cursor:pointer;background:#f9fbe7}
#drop:hover{background:#f0f4c3}#preview{max-width:100%;border-radius:8px;margin-top:12px;display:none}
button{background:#2e7d32;color:#fff;border:0;border-radius:8px;padding:12px 18px;font-size:16px;cursor:pointer;width:100%;margin-top:12px}
button:disabled{background:#9e9e9e}.pill{display:inline-block;background:#e8f5e9;color:#1b5e20;border-radius:20px;padding:4px 12px;margin:2px;font-size:13px}
.conf{background:#eee;border-radius:6px;height:10px;overflow:hidden}.conf>div{background:#2e7d32;height:100%}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{border-bottom:1px solid #eee;padding:8px;text-align:left}
.warn{background:#fff8e1;border-left:4px solid #ffa000;padding:10px 14px;font-size:13px;border-radius:0 8px 8px 0}
footer{text-align:center;color:#757575;font-size:12px;padding:20px}
</style></head><body>
<header><h1>🌾 Solar Smart Light Trap — Pest Detection</h1><p>Upload trap image → insect name + IPM / pesticide advice</p></header>
<main>
<div class="card">
<h3>1. Upload insect image</h3>
<div id="drop">📷 Click or drag & drop image here<br><small>JPG / PNG from trap camera</small><input type="file" id="file" accept="image/*" hidden></div>
<img id="preview">
<button id="btn" disabled>Predict Pest</button>
<div id="status" style="margin-top:10px;font-size:14px;color:#555"></div>
</div>
<div class="card">
<h3>2. Result</h3>
<div id="result"><p style="color:#888">No image yet.</p></div>
</div>
</main>
<div style="max-width:900px;margin:0 auto;padding:0 16px"><div class="card"><h3>Recommendation</h3><div id="reco"><p style="color:#888">Upload to see cultural, biological & chemical options.</p></div>
<div class="warn" style="margin-top:12px">DISCLAIMER_PLACEHOLDER</div>
</div></div>
<footer>EfficientNet-B1 (9 pests) • runs on laptop • ESP32 trap sends image here on IR trigger</footer>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),prev=document.getElementById('preview'),btn=document.getElementById('btn'),status=document.getElementById('status');
drop.onclick=()=>file.click();
['dragover','dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();if(e=='drop'&&ev.dataTransfer.files[0]){file.files=ev.dataTransfer.files;show()}}));
file.onchange=show;
function show(){const f=file.files[0];if(!f)return;prev.src=URL.createObjectURL(f);prev.style.display='block';btn.disabled=false;status.textContent=f.name}
btn.onclick=async()=>{const f=file.files[0];if(!f)return;btn.disabled=true;status.textContent='Predicting...';
const fd=new FormData();fd.append('image',f);
const r=await fetch('/predict',{method:'POST',body:fd});const j=await r.json();btn.disabled=false;status.textContent='';
if(j.error){document.getElementById('result').innerHTML='<p style=color:red>'+j.error+'</p>';return}
let h=`<h2 style="margin:0">${j.common_name}</h2><p>Top match: <b>${j.best}</b> — ${j.top1_conf}% confidence</p>`;
j.top3.forEach(t=>{h+=`<div>${t.class} — ${t.confidence}%<div class=conf><div style="width:${t.confidence}%"></div></div></div>`});
h+=`<p><b>Damage:</b> ${j.damage||'-'}</p>`;document.getElementById('result').innerHTML=h;
let k=`<h4>🛡 Cultural / IPM (do first)</h4>`+(j.ipm_cultural||[]).map(x=>`<span class=pill>${x}</span>`).join(' ');
k+=`<h4>🐞 Biological</h4>`+(j.biological||[]).map(x=>`<span class=pill>${x}</span>`).join(' ');
k+=`<h4>🧪 Chemical options</h4><table><tr><th>Product</th><th>Dose</th><th>Note</th></tr>`+(j.chemical||[]).map(c=>`<tr><td>${c.product}</td><td>${c.dose}</td><td>${c.note||''}</td></tr>`).join('')+`</table>`;
document.getElementById('reco').innerHTML=k;}
</script></body></html>""".replace("DISCLAIMER_PLACEHOLDER", DISCLAIMER)

@app.route("/")
def index():
    return HTML

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files["image"]
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400
    best, top3, info = predict_pil(img)
    return jsonify({
        "best": best,
        "top1_conf": top3[0]["confidence"],
        "top3": top3,
        "common_name": info.get("common_name", best),
        "damage": info.get("damage", ""),
        "ipm_cultural": info.get("ipm_cultural", []),
        "biological": info.get("biological", []),
        "chemical": info.get("chemical", []),
    })

if __name__ == "__main__":
    get_model()  # warm up, fail fast if best_model.pt missing
    app.run(host="127.0.0.1", port=5000, debug=False)

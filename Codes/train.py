"""
Pest classifier training - EfficientNet-B1 transfer learning (PyTorch + timm).
Research basis: on the BAU-Insectv2 pest dataset, EfficientNet-B1 reached 99.57%
test accuracy with Adam, beating ResNet101V2 (96.55%), InceptionV3 (98.25%) and
Inception-ResNetV1 (98.44%), at only 0.6B FLOPs / 30MB / 3.1ms GPU img.
So EfficientNet-B1 = best accuracy/efficiency trade-off for 9-class field pest ID.
"""
import json, os, random, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------- config ----------------
BASE = Path(__file__).parent
TRAIN_DIR = Path(r"C:\Users\Monika\Desktop\Agriculture Project\BAU-Insectv2\extracted_folder\pest\train")
TEST_DIR  = Path(r"C:\Users\Monika\Desktop\Agriculture Project\BAU-Insectv2\extracted_folder\pest\test")
OUT_DIR = BASE / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "efficientnet_b1"   # timm name; pretrained on ImageNet
IMG_SIZE = 240                   # EfficientNet-B1 native resolution
BATCH = 32
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.15
SEED = 42
PATIENCE = 5
NUM_WORKERS = 2

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    train_tf, eval_tf = build_transforms()
    full = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    classes = full.classes
    print("Classes:", classes)
    (OUT_DIR / "classes.json").write_text(json.dumps(classes, indent=2))

    # stratified train/val split
    targets = [s[1] for s in full.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_idx, va_idx = next(sss.split(np.zeros(len(targets)), targets))
    # val set must use eval transforms -> separate ImageFolder instance
    full_eval = datasets.ImageFolder(str(TRAIN_DIR), transform=eval_tf)
    train_set = Subset(full, tr_idx)
    val_set = Subset(full_eval, va_idx)
    test_set = datasets.ImageFolder(str(TEST_DIR), transform=eval_tf)
    print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # class weights for mild imbalance (stem_borer 181 vs mosquito 295)
    counts = np.bincount(targets)
    w = len(targets) / (len(classes) * counts)
    class_w = torch.tensor(w, dtype=torch.float32).to(device)
    print("Class counts:", dict(zip(classes, counts.tolist())))

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = 0.0; bad = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def run_epoch(loader, train: bool):
        model.train(train)
        tot_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(x)
                    loss = criterion(out, y)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        return tot_loss / n, correct / n

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        va_loss, va_acc = run_epoch(val_loader, False)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print(f"Epoch {ep}/{EPOCHS}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")
        if va_acc > best_val:
            best_val = va_acc; bad = 0
            torch.save({"model": MODEL_NAME, "classes": classes, "img_size": IMG_SIZE,
                        "state_dict": model.state_dict()}, OUT_DIR / "best_model.pt")
            print(f"  -> saved best ({best_val:.4f})")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep} (best val {best_val:.4f})")
                break
    print(f"Training time: {(time.time()-t0)/60:.1f} min. Best val acc: {best_val:.4f}")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    # ---- final test evaluation with best weights ----
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().tolist()
            all_p += p; all_t += y.tolist()
    acc = accuracy_score(all_t, all_p)
    print(f"\nTEST accuracy: {acc:.4f}")
    print(classification_report(all_t, all_p, target_names=classes, digits=4))
    print("Confusion matrix:\n", confusion_matrix(all_t, all_p))
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(
        {"test_acc": acc, "report": classification_report(all_t, all_p, target_names=classes, output_dict=True),
         "confusion_matrix": confusion_matrix(all_t, all_p).tolist()}, indent=2))

    # training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].legend()
        ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["val_acc"], label="val")
        ax[1].set_title("Accuracy"); ax[1].legend()
        fig.suptitle(f"EfficientNet-B1 pest classifier (best val {best_val:.3f}, test {acc:.3f})")
        fig.tight_layout(); fig.savefig(OUT_DIR / "curves.png", dpi=120)
        print("Saved curves.png")
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    main()

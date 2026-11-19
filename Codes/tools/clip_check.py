"""
CLIP zero-shot helper: used to (a) verify that candidate cartoon images really depict
the assigned class, and (b) verify that 'negative' images contain no insect.
Also used as a sanity check on the real training images.

Model: open_clip ViT-B-32 / laion2b_s34b_b79k (zero-shot ImageNet-classified CLIP).
"""
import torch
import open_clip
from PIL import Image

_MODEL = None
_PREP = None

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLASS_PROMPTS = {
    "aphids":      ["a cartoon drawing of aphids", "an illustration of aphids on a plant",
                    "a photo of aphids", "a picture of aphids sucking plant sap"],
    "armyworm":    ["a cartoon drawing of an armyworm caterpillar", "an illustration of a caterpillar with stripes",
                    "a photo of an armyworm caterpillar", "a picture of a hairy caterpillar"],
    "beetle":      ["a cartoon drawing of a beetle", "an illustration of a beetle insect",
                    "a photo of a beetle", "a picture of a hard shelled beetle"],
    "bollworm":    ["a cartoon drawing of a bollworm caterpillar", "an illustration of a bollworm larva",
                    "a photo of a bollworm caterpillar", "a picture of a fat caterpillar on a plant"],
    "grasshopper": ["a cartoon drawing of a grasshopper", "an illustration of a grasshopper",
                    "a photo of a grasshopper", "a picture of a grasshopper with long hind legs"],
    "mites":       ["a cartoon drawing of a spider mite", "an illustration of a red spider mite",
                    "a photo of spider mites on a leaf", "a picture of tiny mites with webbing"],
    "mosquito":    ["a cartoon drawing of a mosquito", "an illustration of a mosquito",
                    "a photo of a mosquito", "a picture of a mosquito with long legs"],
    "sawfly":      ["a cartoon drawing of a sawfly larva", "an illustration of a sawfly caterpillar",
                    "a photo of a sawfly larva", "a picture of a sawfly insect"],
    "stem_borer":  ["a cartoon drawing of a stem borer caterpillar", "an illustration of a stem borer larva boring a stem",
                    "a photo of a stem borer caterpillar", "a picture of a caterpillar boring into a plant stem"],
}

NEGATIVE_PROMPTS = [
    "a photo of a fruit", "a photo of a piece of furniture", "a photo of a shoe",
    "a photo of a landscape", "a photo of a person", "a photo of a car",
    "a photo of a machine tool", "a photo of a building", "a photo of food on a plate",
    "a photo of a tree", "a photo of a book", "a photo of a computer",
]


def get_clip(device="cuda"):
    global _MODEL, _PREP
    if _MODEL is None:
        model, _, prep = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=None
        )
        model = model.to(device).eval()
        _MODEL, _PREP = model, prep
    return _MODEL, _PREP


@torch.no_grad()
def clip_scores(images, prompts, device="cuda"):
    """Return (n_images, n_prompts) cosine-similarity matrix."""
    model, prep = get_clip(device)
    # average text embedding per prompt group
    tok = open_clip.get_tokenizer("ViT-B-32")
    txt = []
    for p in prompts:
        t = tok(p).to(device)
        with torch.no_grad():
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
        txt.append(e.mean(0))
    txt = torch.stack(txt)
    txt = txt / txt.norm(dim=-1, keepdim=True)

    feats = []
    for im in images:
        x = prep(im.convert("RGB")).unsqueeze(0).to(device)
        f = model.encode_image(x)
        feats.append(f / f.norm(dim=-1, keepdim=True))
    imf = torch.cat(feats)
    return (imf @ txt.T).float().cpu()


@torch.no_grad()
def classify_zero_shot(img, device="cuda"):
    """Return {class: mean_similarity} for the 9 pest classes."""
    classes = list(CLASS_PROMPTS.keys())
    scores = clip_scores([img], [CLASS_PROMPTS[c] for c in classes], device)[0]
    return {c: float(s) for c, s in zip(classes, scores)}


@torch.no_grad()
def insect_score(img, device="cuda"):
    """Max similarity over the 9 insect class prompts and the negative prompts."""
    classes = list(CLASS_PROMPTS.keys())
    s_ins = clip_scores([img], [CLASS_PROMPTS[c] for c in classes], device)[0]
    s_neg = clip_scores([img], NEGATIVE_PROMPTS, device)[0]
    return float(s_ins.max()), float(s_neg.max())

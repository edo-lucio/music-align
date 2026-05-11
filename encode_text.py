"""Generate per-frame captions with BLIP, encode with a sentence-transformer.

Outputs:
    cache/captions.json   — list of {id, caption}, easy to grep for label leakage
    cache/embeds/text_minilm.npz  — N x D text embeddings aligned to the manifest order

Caveat: captions come from BLIP looking at the frame, so the text modality is
biased toward vision. Grep cache/captions.json for instrument names to gauge
how much the class label leaks via the captioner; report this honestly.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)

CAPTIONER = "Salesforce/blip-image-captioning-base"
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
INPUTS = Path("cache/inputs")
EMBEDS = Path("cache/embeds")
CAPTIONS_OUT = Path("cache/captions.json")


def _pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")).sum().item()
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = _pick_device()


def caption_frames(manifest: list[dict]) -> list[str]:
    proc = BlipProcessor.from_pretrained(CAPTIONER)
    model = BlipForConditionalGeneration.from_pretrained(CAPTIONER).to(DEVICE).eval()
    captions = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(it["frame"]).convert("RGB")
            inputs = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_length=30, num_beams=3)
            captions.append(proc.decode(out[0], skip_special_tokens=True).strip())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return captions


def encode_text(texts: list[str]) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    model = AutoModel.from_pretrained(TEXT_ENCODER).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            # mean-pool with attention mask (standard sentence-transformer pooling)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(embs).astype(np.float32)


def main() -> None:
    EMBEDS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUTS / "manifest.json").read_text())

    if CAPTIONS_OUT.exists():
        print(f"loading cached captions from {CAPTIONS_OUT}")
        records = json.loads(CAPTIONS_OUT.read_text())
        cap_by_id = {r["id"]: r["caption"] for r in records}
        captions = [cap_by_id[it["id"]] for it in manifest]
    else:
        print(f"captioning {len(manifest)} frames with BLIP ...")
        captions = caption_frames(manifest)
        CAPTIONS_OUT.write_text(json.dumps(
            [{"id": it["id"], "instrument": it["instrument"], "caption": c}
             for it, c in zip(manifest, captions)],
            indent=2,
        ))
        print(f"  wrote {CAPTIONS_OUT}")

    out = EMBEDS / "text_minilm.npz"
    if out.exists():
        print(f"skip {out.name}")
    else:
        print("encoding captions with sentence-transformer ...")
        X = encode_text(captions)
        labels = np.array([it["instrument"] for it in manifest])
        ids = np.array([it["id"] for it in manifest])
        np.savez(out, X=X, labels=labels, ids=ids)
        print(f"  -> {X.shape}")

    # quick leakage diagnostic
    from collections import Counter
    instruments = sorted({it["instrument"] for it in manifest})
    hits = Counter()
    for c, it in zip(captions, manifest):
        for name in instruments:
            if name.replace("_", " ") in c.lower() or name in c.lower():
                hits[it["instrument"]] += int(name == it["instrument"])
    total = len(manifest)
    leaked = sum(hits.values())
    print(f"\nleakage check: {leaked}/{total} captions contain their own instrument name "
          f"({100*leaked/total:.1f}%)")
    print("inspect cache/captions.json to review.")


if __name__ == "__main__":
    main()

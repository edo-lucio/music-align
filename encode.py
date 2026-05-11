"""Encode cached frames and audio clips with each pretrained model.

Saves one .npz per (modality, model_name) into cache/embeds/.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, ClapModel, ClapProcessor

# Vision ladder: clean small -> giant param scaling.
VISION_MODELS = {
    "dinov2-small": "facebook/dinov2-small",   # 21M
    "dinov2-base":  "facebook/dinov2-base",    # 86M
    "dinov2-large": "facebook/dinov2-large",   # 300M
    # "dinov2-giant": "facebook/dinov2-giant", # 1.1B — uncomment if you have a recent GPU
}

# Audio ladder: CLAP variants — note this is less of a clean param ladder than DINOv2.
# Scaling claim is mostly testable on the vision side.
AUDIO_MODELS = {
    "clap-unfused": "laion/clap-htsat-unfused",
    "clap-fused":   "laion/clap-htsat-fused",
    "clap-larger":  "laion/larger_clap_general",
    "clap-music":   "laion/larger_clap_music",
}

def _pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")).sum().item()
        return "cuda"
    except Exception as e:
        print(f"CUDA unusable ({e.__class__.__name__}); falling back to CPU")
        return "cpu"


DEVICE = _pick_device()
print(f"device = {DEVICE}")
INPUTS = Path("cache/inputs")
EMBEDS = Path("cache/embeds")


def encode_vision(model_id: str, manifest: list[dict]) -> np.ndarray:
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(it["frame"]).convert("RGB")
            x = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model(**x)
            pooled = getattr(out, "pooler_output", None)
            f = pooled if pooled is not None else out.last_hidden_state[:, 0]
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def encode_audio(model_id: str, manifest: list[dict]) -> np.ndarray:
    proc = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            wav, sr = sf.read(it["audio"])
            x = proc(audio=wav, sampling_rate=sr, return_tensors="pt").to(DEVICE)
            out = model.get_audio_features(**x)
            # transformers 5.x returns BaseModelOutputWithPooling, not a tensor.
            f = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def main() -> None:
    EMBEDS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    labels = np.array([it["instrument"] for it in manifest])
    ids = np.array([it["id"] for it in manifest])

    for name, mid in VISION_MODELS.items():
        out = EMBEDS / f"vision_{name}.npz"
        if out.exists():
            print(f"skip {out.name}")
            continue
        print(f"encoding {name} ...")
        X = encode_vision(mid, manifest)
        np.savez(out, X=X, labels=labels, ids=ids)
        print(f"  -> {X.shape}")

    for name, mid in AUDIO_MODELS.items():
        out = EMBEDS / f"audio_{name}.npz"
        if out.exists():
            print(f"skip {out.name}")
            continue
        print(f"encoding {name} ...")
        X = encode_audio(mid, manifest)
        np.savez(out, X=X, labels=labels, ids=ids)
        print(f"  -> {X.shape}")


if __name__ == "__main__":
    main()

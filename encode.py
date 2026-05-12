"""Encode cached frames and audio clips with each pretrained model.

Per-model dispatch handles the different APIs (DINOv2 vs CLIP vs MAE on the
vision side; CLAP vs MERT on the audio side). Output: one .npz per
(modality, model_name) into cache/embeds/.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# --- registry ---------------------------------------------------------------
# Each entry: name -> {hf_id, type, ...optional}. "type" picks the encode_fn.

VISION_ENCODERS = {
    "dinov2-small": {"hf_id": "facebook/dinov2-small",         "type": "auto_pooler"},  # 21M, SSL
    "dinov2-base":  {"hf_id": "facebook/dinov2-base",          "type": "auto_pooler"},  # 86M, SSL
    "dinov2-large": {"hf_id": "facebook/dinov2-large",         "type": "auto_pooler"},  # 300M, SSL
    "clip-base":    {"hf_id": "openai/clip-vit-base-patch32",  "type": "clip"},          # 88M, vision-text
    "clip-large":   {"hf_id": "openai/clip-vit-large-patch14", "type": "clip"},          # 304M, vision-text
    "vit-mae-base": {"hf_id": "facebook/vit-mae-base",         "type": "mae"},           # 86M, SSL (masked AE)
}

AUDIO_ENCODERS = {
    "clap-unfused": {"hf_id": "laion/clap-htsat-unfused",  "type": "clap"},
    "clap-fused":   {"hf_id": "laion/clap-htsat-fused",    "type": "clap"},
    "clap-larger":  {"hf_id": "laion/larger_clap_general", "type": "clap"},
    "clap-music":   {"hf_id": "laion/larger_clap_music",   "type": "clap"},
    "mert-95m":     {"hf_id": "m-a-p/MERT-v1-95M",         "type": "mert", "sr": 24000},  # music SSL
    "mert-330m":    {"hf_id": "m-a-p/MERT-v1-330M",        "type": "mert", "sr": 24000},  # music SSL, larger
}


# --- device -----------------------------------------------------------------

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


# --- helpers ----------------------------------------------------------------

def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return wav, sr_in
    import librosa  # lazy import
    return librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=sr_out), sr_out


# --- vision encoders --------------------------------------------------------

def _encode_vision_auto_pooler(hf_id, manifest):
    """DINOv2-style: AutoImageProcessor + AutoModel, take pooler_output (CLS)."""
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
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


def _encode_vision_clip(hf_id, manifest):
    """CLIP: contrastive projection head -> (B, projection_dim) image_embeds."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    proc = CLIPImageProcessor.from_pretrained(hf_id)
    model = CLIPVisionModelWithProjection.from_pretrained(hf_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(it["frame"]).convert("RGB")
            x = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model(**x)
            f = out.image_embeds  # (1, projection_dim)
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def _encode_vision_mae(hf_id, manifest):
    """ViT-MAE: load weights into a plain ViT (no masking), take CLS."""
    from transformers import ViTModel
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = ViTModel.from_pretrained(hf_id, add_pooling_layer=False).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(it["frame"]).convert("RGB")
            x = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model(**x)
            f = out.last_hidden_state[:, 0]  # CLS token
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


VISION_DISPATCH = {
    "auto_pooler": _encode_vision_auto_pooler,
    "clip":        _encode_vision_clip,
    "mae":         _encode_vision_mae,
}


# --- audio encoders ---------------------------------------------------------

def _encode_audio_clap(hf_id, manifest, **cfg):
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained(hf_id)
    model = ClapModel.from_pretrained(hf_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            wav, sr = sf.read(it["audio"])
            x = proc(audio=wav, sampling_rate=sr, return_tensors="pt").to(DEVICE)
            out = model.get_audio_features(**x)
            f = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def _encode_audio_mert(hf_id, manifest, sr=24000, **cfg):
    """MERT: music-specific SSL. Wants 24kHz mono; mean-pool last hidden state."""
    from transformers import Wav2Vec2FeatureExtractor
    proc = Wav2Vec2FeatureExtractor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            wav, sr_in = sf.read(it["audio"])
            if wav.ndim > 1:
                wav = wav.mean(axis=1)  # ensure mono
            wav, _ = _resample(wav, sr_in, sr)
            x = proc(wav, sampling_rate=sr, return_tensors="pt").to(DEVICE)
            out = model(**x)
            f = out.last_hidden_state.mean(dim=1)  # mean-pool over time
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


AUDIO_DISPATCH = {
    "clap": _encode_audio_clap,
    "mert": _encode_audio_mert,
}


# --- driver -----------------------------------------------------------------

def _encode_and_save(name, cfg, dispatch, manifest, modality, labels, ids):
    out = EMBEDS / f"{modality}_{name}.npz"
    if out.exists():
        print(f"skip {out.name}")
        return
    fn = dispatch[cfg["type"]]
    print(f"encoding {modality} · {name} ({cfg['hf_id']}) ...")
    extra = {k: v for k, v in cfg.items() if k not in ("hf_id", "type")}
    X = fn(cfg["hf_id"], manifest, **extra)
    if X.ndim != 2 or X.shape[0] != len(manifest):
        raise RuntimeError(
            f"{modality}/{name} produced bad shape {X.shape}; expected (N={len(manifest)}, D)"
        )
    np.savez(out, X=X, labels=labels, ids=ids)
    print(f"  -> {X.shape}")


def main() -> None:
    EMBEDS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    labels = np.array([it["instrument"] for it in manifest])
    ids = np.array([it["id"] for it in manifest])

    for name, cfg in VISION_ENCODERS.items():
        try:
            _encode_and_save(name, cfg, VISION_DISPATCH, manifest, "vision", labels, ids)
        except Exception as e:
            print(f"  ! vision/{name} failed: {e.__class__.__name__}: {e}")

    for name, cfg in AUDIO_ENCODERS.items():
        try:
            _encode_and_save(name, cfg, AUDIO_DISPATCH, manifest, "audio", labels, ids)
        except Exception as e:
            print(f"  ! audio/{name} failed: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()

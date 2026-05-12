"""Extract one random frame + 1 audio clip per video.

Reads videos from `video/video/*.mp4` and looks up class labels in
`vggsound.csv` (no header; columns: ytid, start_seconds, label, split).
Filenames are `{ytid}_{start:06d}.mp4`, which is the lookup key.

Output: cache/inputs/<video_stem>/{frame.jpg, audio.wav} + manifest.json
Manifest entry: {id, instrument, frame, audio}. `instrument` carries the
VGGSound class label (kept under that key for back-compat with the rest
of the pipeline, even though VGGSound labels are not always instruments).
"""
import csv
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

VIDEO_DIR = Path("video/video")
VGGSOUND_CSV = Path("vggsound.csv")
OUT_DIR = Path("cache/inputs")
CAP = 22                   # per-class cap; most VGGSound classes have far fewer
AUDIO_SR = 48000           # CLAP expects 48kHz
CLIP_SECONDS = 5.0
SEED = 0


def load_label_lookup(csv_path: Path) -> dict[str, str]:
    """Map `{ytid}_{start:06d}` -> class label."""
    out = {}
    with csv_path.open() as f:
        for row in csv.reader(f):
            ytid, start, label, _split = row[0], int(row[1]), row[2], row[3]
            out[f"{ytid}_{start:06d}"] = label
    return out


def duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.strip())


def extract(video: Path, frame: Path, audio: Path, rng: random.Random) -> None:
    dur = duration(video)
    t = rng.uniform(0.5, max(0.6, dur - CLIP_SECONDS - 0.5))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
         "-i", str(video), "-frames:v", "1", str(frame)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
         "-i", str(video), "-t", str(CLIP_SECONDS),
         "-ac", "1", "-ar", str(AUDIO_SR), str(audio)],
        check=True,
    )


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_of = load_label_lookup(VGGSOUND_CSV)

    # Group videos on disk by class label
    by_class: dict[str, list[Path]] = defaultdict(list)
    for v in sorted(VIDEO_DIR.glob("*.mp4")):
        label = label_of.get(v.stem)
        if label is None:
            print(f"skip {v.stem}: no row in vggsound.csv")
            continue
        by_class[label].append(v)

    manifest = []
    for label, vids in sorted(by_class.items()):
        picked = rng.sample(vids, min(CAP, len(vids)))
        for v in picked:
            d = OUT_DIR / v.stem
            d.mkdir(exist_ok=True)
            frame, audio = d / "frame.jpg", d / "audio.wav"
            if not frame.exists() or not audio.exists():
                try:
                    extract(v, frame, audio, rng)
                except subprocess.CalledProcessError as e:
                    print(f"skip {v.stem}: {e}")
                    continue
            manifest.append({
                "id": v.stem, "instrument": label,
                "frame": str(frame), "audio": str(audio),
            })
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} items across {len(by_class)} classes")


if __name__ == "__main__":
    main()

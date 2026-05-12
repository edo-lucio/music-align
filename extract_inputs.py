"""Extract one random frame + 1s audio clip per video, balanced per instrument.

Output: cache/inputs/<video_stem>/{frame.jpg, audio.wav} + manifest.json
"""
import json
import random
import subprocess
from pathlib import Path

VIDEO_DIR = Path("video")
OUT_DIR = Path("cache/inputs")
CAP = 22                   # min class size (saxophone)
AUDIO_SR = 48000           # CLAP expects 48kHz
CLIP_SECONDS = 5.0
SEED = 0

INSTRUMENTS = [
    "accordion", "acoustic_guitar", "cello", "clarinet", "erhu", "flute",
    "saxophone", "trumpet", "tuba", "violin", "xylophone",
]

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
    manifest = []
    for inst in INSTRUMENTS:
        vids = sorted(VIDEO_DIR.glob(f"{inst}_*.mp4"))
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
                "id": v.stem, "instrument": inst,
                "frame": str(frame), "audio": str(audio),
            })
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} items across {len(INSTRUMENTS)} instruments")


if __name__ == "__main__":
    main()

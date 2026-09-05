#!/usr/bin/env python3
"""CoreCast Stage 2: audio.wav -> transcript via local nemo-speech ASR.

Uses NVIDIA NeMo-Speech.cpp (Parakeet-TDT-0.6B-v3, CUDA) built project-locally.

Usage: python transcribe.py <audio.wav> [--model PATH] [--engine PATH]
              [--device DEV] [--language CODE] [--format text|json]
              [--force] [-v] [--json-progress]

Output: <audio_dir>/transcript.txt (or .json) - plain text with punctuation
        and capitalization, language auto-detected.

Exit codes: 0 ok/skip | 1 engine failure | 2 input missing |
            3 engine/model not found | 4 empty output | 5 usage
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fetch import (CLIReporter, JSONReporter, fmt_dur, human_bytes,
                   probe_audio)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENGINE = PROJECT_ROOT / "bin" / "nemo-speech.exe"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3.q8_0.gguf"
MODELS_DIR = PROJECT_ROOT / "models"


# ------------------------------------------------------------------ utils

def find_engine(explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    cands = [DEFAULT_ENGINE, Path("bin") / "nemo-speech.exe"]
    on_path = shutil.which("nemo-speech")
    if on_path:
        cands.append(Path(on_path))
    for c in cands:
        if c.is_file():
            return c
    return None


def find_model(explicit=None):
    if explicit:
        m = Path(explicit)
        return m if m.is_file() else None
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    if MODELS_DIR.is_dir():
        gs = sorted(MODELS_DIR.glob("*.gguf"))
        if gs:
            return gs[0]
    return None


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="transcribe.py",
                                 description="CoreCast Stage 2: audio -> transcript (local ASR)")
    ap.add_argument("audio", help="input audio file (16 kHz mono WAV)")
    ap.add_argument("--model", help="ASR GGUF path (default: models/parakeet-tdt-0.6b-v3.q8_0.gguf)")
    ap.add_argument("--engine", help="nemo-speech executable (default: bin/nemo-speech.exe)")
    ap.add_argument("--device", help="auto, cpu, cuda[:N], metal, vulkan[:N] (default: auto)")
    ap.add_argument("--language", help="language code or prompt (default: auto-detect)")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe even if the transcript exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json-progress", action="store_true",
                    help="machine-readable events (one JSON line each)")
    args = ap.parse_args(argv)

    # preflight
    audio = Path(args.audio)
    if not audio.is_file():
        print(f"ERROR: audio file not found: {audio} (exit 2)")
        return 2
    engine = find_engine(args.engine)
    if engine is None:
        print("ERROR: nemo-speech executable not found (build it into bin/ "
              "or pass --engine) (exit 3)")
        return 3
    model = find_model(args.model)
    if model is None:
        print("ERROR: ASR model GGUF not found (download into models/ "
              "or pass --model) (exit 3)")
        return 3

    rep = JSONReporter() if args.json_progress else CLIReporter(sys.stdout.isatty())

    info = probe_audio(audio)
    dur_s = info.get("duration_s") if info else None
    out = audio.with_name(f"transcript.{'json' if args.format == 'json' else 'txt'}")

    # existing transcript -> skip
    if out.is_file() and out.stat().st_size > 0 and not args.force:
        rep.status(f"[3/3] verify    -> [PASS] existing transcript, skipped "
                   f"transcription")
        rep.status(f"SKIP {out} | {human_bytes(out.stat().st_size)}")
        return 0
    if args.force and out.is_file():
        out.unlink()

    rep.status(f"[1/3] info      -> audio={fmt_dur(dur_s)}, "
               f"device={args.device or 'auto'}, "
               f"model={model.name}, format={args.format}")
    rep.status(f"[2/3] transcribe -> {engine.name} (offline ASR, CUDA) ...")

    t1 = time.perf_counter()
    cmd = [str(engine), "transcribe", str(audio),
           "--model", str(model),
           "-f", args.format,
           "-o", str(out)]
    if args.device:
        cmd += ["--device", args.device]
    if args.language:
        cmd += ["--language", args.language]
    if args.verbose:
        cmd += ["--verbose"]
    # generous timeout, scaled by audio duration (engine runs ~50x realtime)
    timeout = max(120, (dur_s or 0) / 20 + 60)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        rep.status(f"ERROR: transcription timed out after {timeout:.0f}s (exit 1)")
        return 1
    dt = time.perf_counter() - t1

    if r.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        tail = (r.stderr.strip().splitlines() or ["engine failed"])[-1]
        rep.status(f"ERROR: {tail} (exit 1)")
        return 1

    # [3/3] verify
    size = out.stat().st_size
    words = None
    if args.format == "text":
        words = len(out.read_text(encoding="utf-8", errors="replace").split())
    rep.status(f"[3/3] verify    -> [PASS] {human_bytes(size)}"
               + (f", {words} words" if words is not None else ""))
    rep.status(f"OK  {out} | {fmt_dur(dur_s)} audio | {dt:.1f}s wall")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# CoreCast

Pipeline: public web video (YouTube / VK / Rutube) -> structured summary text.

## What it does

1. Downloads only the audio track of a video given its URL (yt-dlp).
2. Transcribes it locally on GPU (NeMo-Speech.cpp, Parakeet-TDT-0.6B-v3).
3. Summarizes the transcript with the DeepSeek API (deepseek-v4-flash):
   the major ideas, in the language of the video.

## Status

Stage-based development, each stage is discussed and approved before
implementation.

- [x] Stage 1 plan (URL -> audio WAV): docs/stage1-plan.md
- [x] Stage 1 implementation
- [x] Stage 2 (audio -> transcript)
- [x] Stage 3 (transcript -> summary)
- [x] Stage 4 (GUI): pywebview window (ui/index.html), pipeline.py runner
- [ ] Stage 5 (packaging): PyInstaller exe (deferred)

## GUI

Run by double-clicking `CoreCast.lnk` (pythonw + WebView2, no console), or:

```
python gui.py
```

The GUI wraps the three CLI stages via `pipeline.py`, mapping progress into
one continuous 0-100% bar (internal split: download 50 / transcribe 40 /
summarize 10). Run artifacts go to `%LOCALAPPDATA%/CoreCast/run/`.
Regenerate the shortcut after moving the project: `make_shortcut.ps1`.

## Requirements

- Python 3.11, ffmpeg/ffprobe on PATH
- NVIDIA GPU (CUDA) for transcription
- DeepSeek API key in `.env`: `DEEPSEEK_API_KEY` (platform.deepseek.com)

## Usage

```
python fetch.py <url> [--out-dir DIR] [--force] [-v] [--json-progress]
python transcribe.py <audio.wav> [--model PATH] [--device cuda] [--format text|json] [--force]
python summarize.py <transcript.txt> [--model M] [--force] [--json-progress]
```

- `fetch.py` - URL -> 16 kHz mono WAV (yt-dlp; progress bar / --json-progress)
- `transcribe.py` - WAV -> transcript (nemo-speech, Parakeet-TDT-0.6B-v3,
  CUDA, ~50x realtime on an RTX 4060). `--force`: re-transcribe
- `summarize.py` - transcript -> major-ideas summary (DeepSeek API,
  deepseek-v4-pro by default, single-pass up to ~200k chars of
  transcript). `--model deepseek-v4-flash`: ~3x faster, leaner output.

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
- [ ] Stage 4 (hardening / packaging)

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
  deepseek-v4-flash by default, single-pass up to ~200k chars of
  transcript). `--model`: pick another DeepSeek model (e.g.
  deepseek-v4-pro for maximum detail).

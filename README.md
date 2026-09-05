# CoreCast

Local pipeline: public web video (YouTube / VK / Rutube) -> standalone summary text.

## What it does

1. Downloads only the audio track of a video given its URL.
2. Transcribes it locally (Whisper large-v3, GPU).
3. Summarizes the transcript into a pure standalone text of the major
   thoughts (local LLM) - no metadata, no timestamps, no references to
   the source video.

Everything runs on the local machine; no cloud APIs.

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
- (later stages) NVIDIA GPU, faster-whisper, llama.cpp

## Usage

```
python fetch.py <url> [--out-dir DIR] [--force] [-v] [--json-progress]
python transcribe.py <audio.wav> [--model PATH] [--device cuda] [--format text|json] [--force]
python summarize.py <transcript.txt> [--backend groq|gigachat] [--model M] [--force] [--json-progress]
```

- `fetch.py` - URL -> 16 kHz mono WAV (yt-dlp; progress bar / --json-progress)
- `transcribe.py` - WAV -> transcript (nemo-speech, Parakeet-TDT-0.6B-v3,
  CUDA, ~50x realtime on an RTX 4060). `--force`: re-transcribe
- `summarize.py` - transcript -> standalone summary (online LLM).
  `groq` backend: free tier, set `GROQ_API_KEY` in `.env`;
  `gigachat`: `GIGACHAT_CLIENT_ID` + `GIGACHAT_API_KEY` (Sber freemium).
  Long transcripts are chunked (map-reduce) and rate limits self-paced.

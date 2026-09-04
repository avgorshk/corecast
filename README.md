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
- [ ] Stage 1 implementation
- [ ] Stage 2 (audio -> transcript)
- [ ] Stage 3 (transcript -> summary)
- [ ] Stage 4 (hardening / packaging)

## Requirements

- Python 3.11, ffmpeg/ffprobe on PATH
- (later stages) NVIDIA GPU, faster-whisper, llama.cpp

## Usage (planned)

```
python fetch.py <url>
```

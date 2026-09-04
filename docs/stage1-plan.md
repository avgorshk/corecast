# audio2text - Stage 1: URL -> audio (WAV)

Status: PLAN (awaiting approval)
Date: 2026-09-04

## 0. Product requirement (drives all stages)

Final deliverable = PURE standalone summary text of the video's major
thoughts. No metadata, no title/URL, no timestamps, no references to
"the video" or its author. The summary must read like self-contained text.

Pipeline artifacts on disk (intermediates, wiped by --clean in Stage 4):

```
data/<video_id>/audio.wav       Stage 1 (this plan)
data/<video_id>/transcript.txt  Stage 2
data/<video_id>/summary.txt     Stage 3 (final output, pure text)
```

No metadata files are persisted. Metadata is fetched in-memory only for
verification and error handling.

## 1. Scope

Given a URL to a public video on YouTube / VK / Rutube:

- download ONLY the audio track; convert to 16 kHz mono PCM WAV -> audio.wav
- verify in-memory (ffprobe + duration sanity), report [PASS]/[FAIL], exit code
- idempotent re-runs: existing audio.wav is re-verified, download skipped ([SKIP])

Out of scope (later stages): transcription, summarization, config file,
--clean flag, playlists, live streams, login-walled videos, multi-video batches.

## 2. Project layout

```
D:\Projects\audio2text\
├── .venv\                   # regular venv (python -m venv .venv)
├── requirements.txt         # pinned deps (yt-dlp==<version>)
├── fetch.py                 # Stage 1 entry point (later wrapped by a2t CLI)
├── docs\stage1-plan.md      # this file
└── data\
    └── <video_id>\
        └── audio.wav        # 16 kHz mono PCM s16le
```

## 3. fetch.py design

### CLI contract

```
python fetch.py <url> [--out-dir DIR] [--force] [-v]
```

Default out-dir: `data/` relative to the project root.

### Algorithm

```
0. preflight: python >= 3.11, ffmpeg + ffprobe on PATH (exit 3 if missing)
1. yt-dlp -J <url>  (in-memory: id, title, duration, is_live, age_limit)
2. availability checks:
   - is_live == true            -> refuse (exit 2, recorded streams are OK)
   - age_limit > 0 / restricted -> clean error (exit 1)
3. if data/<id>/audio.wav exists AND not --force
   -> re-verify with ffprobe (ms) -> [SKIP], exit 0
4. download + convert (one command):
   yt-dlp -f "bestaudio/best" -x --audio-format wav
          --postprocessor-args "ffmpeg:-ar 16000 -ac 1"
          -o "audio.%(ext)s" --no-playlist -q --no-warnings <url>
   (cwd = data/<id>/)
5. verify with ffprobe (in-memory checks, see 4.)
6. report: duration, audio size, wall time ([PASS]/[FAIL] + exit code)
```

### Console output (ASCII only, no metadata in the report)

```
[1/4] info      -> duration=1457s, live=no            (in-memory)
[2/4] audio     -> bestaudio (opus) 18.9 MB, 31.2s
[3/4] convert   -> audio.wav 16000 Hz mono, 44.5 MiB
[4/4] verify    -> [PASS] codec/sample_rate/channels/duration
OK  data/4eiQNRcoaWc | 1457s | 44.5 MiB | 46s wall
```

Re-run of the same URL prints `[SKIP]` and exits 0.

### Exit codes

| code | meaning |
|---|---|
| 0 | success or [SKIP] |
| 1 | yt-dlp failure (unavailable/private/deleted/geo/age) |
| 2 | live stream refused |
| 3 | ffmpeg/ffprobe missing |
| 4 | verification failed (ffprobe checks or duration mismatch) |
| 5 | usage error (bad args / unparseable URL) |

## 4. Verification rules (in-memory, nothing persisted)

| check | rule | on failure |
|---|---|---|
| codec | pcm_s16le | exit 4 |
| sample_rate | == 16000 | exit 4 |
| channels | == 1 | exit 4 |
| duration | abs(ffprobe - meta) <= 1.0 s | exit 4 (warn only if meta lacks duration) |

## 5. Error taxonomy

- unavailable / private / deleted / geo / age -> yt-dlp exit 1, we print a
  one-line reason, exit 1
- live stream -> refused before download (exit 2)
- VK auth wall (some VK videos require login) -> detect, print hint:
  `--cookies-from-browser chrome` (documented; not implemented in Stage 1)
- ffmpeg missing -> exit 3 with install hint (winget install Gyan.FFmpeg)

## 6. Dependencies & setup

- Python 3.11 (system), venv at .venv, pip install yt-dlp (pinned via
  pip freeze -> requirements.txt)
- ffmpeg 8.1.2 + ffprobe: already installed and on PATH (verified)
- no GPU needed in Stage 1

## 7. Test plan (acceptance criteria)

Positive cases (the 3 real URLs):

| # | URL | platform | expectation |
|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=4eiQNRcoaWc | youtube | PASS, ~24:17 audio |
| 2 | https://rutube.ru/video/aaf306b8abc272d4fab2963d5a799010/ | rutube | PASS (same content as #1) |
| 3 | https://vkvideo.ru/video-203057439_456239522 | vk | PASS, or documented auth-wall fallback |

Negative / behavioral cases:

| # | case | expectation |
|---|---|---|
| 4 | non-existent video URL | clean one-line error, exit 1 |
| 5 | re-run of #1 | [SKIP], exit 0, no re-download |
| 6 | live stream (if one is running at test time) | refused, exit 2; else the is_live branch is exercised by a fixture test |

Acceptance report format: table of platform | duration | audio size |
wall time | [PASS]/[FAIL], plus exact error lines for cases 4-6.

## 8. Test-set observations (valuable for later stages)

- all three URLs are the same video (same channel, same ~24-min episode):
  -> cross-platform audio quality consistency check for Stage 2 ASR
- YouTube version has an official transcript (already extracted):
  -> Stage 2 WER reference without manual work
- VK page requires JS, so its availability is unverified until yt-dlp runs;
  vkvideo.ru is covered by yt-dlp's 'vk' extractor (verified in
  supportedsites.md); owner id 203057439 matches the channel's VK community
  (vk.com/podcasts-203057439), so it is expected to be the same video

## 9. Risks & mitigations

- VK auth wall: try anonymous first; if blocked -> --cookies-from-browser
  (chrome) or replace test video with a known-public one; document result
- YouTube throttling: yt-dlp ships client spoofing; update yt-dlp if needed
- long video edge cases: N/A for Stage 1 test set (~24 min each)

## 10. Implementation checklist

1. create venv, pip install yt-dlp, pin to requirements.txt
2. implement fetch.py (~200 lines, stdlib + subprocess only)
3. run test cases 1-6, record the acceptance table
4. report results; fix anything that fails; then Stage 2 discussion

## 11. Implementation results (2026-09-04)

Acceptance table:

| # | case | result |
|---|---|---|
| 1 | YouTube 4eiQNRcoaWc | [PASS] 10.8 s wall |
| 2 | Rutube aaf306b8... | [PASS] 190.5 s wall (probe-picked tier-1 rendition) |
| 3 | VK -203057439_456239522 | [PASS] 156.0 s wall |
| 4 | non-existent URL | [PASS] exit 1, "This video is unavailable" |
| 5 | re-run of #1 | [PASS] [SKIP], exit 0 |
| 5b | --force #1 | [PASS] true re-download, 10.8 s |
| 6a | live-stream fixture | [PASS] refused before download |
| 6b | real live URL | n/a (none live at test time); fixture covers the branch |

### Deviations from the plan (evidence-driven)

1. Format selection redesigned. Rutube exposes ONLY combined video+audio
   renditions (verified: all 24 formats carry vc=avc1 + ac=mp4a.40.2), so
   `bestaudio/best` fell through to `best` = 1080p ~19.7 Mbps -> a 3.35 GB
   partial download before the first run timed out. New algorithm:
   audio-only formats -> `bestaudio`; combined-only platform -> ffmpeg
   sample-probe the smallest renditions (12 s each) and pick the first
   whose audio is >= 64 kbps.
2. Measured Rutube audio tiers (40 s samples): 144/240/360p = identical
   82 kbps AAC; 476p = 160 kbps; 720/1080p = 232 kbps. Threshold 64 kbps
   accepts tier 1 (transparent for speech ASR); tier 2+ would cost ~5x
   wall time: Rutube serves at ~0.1-0.33 MB/s (measured), i.e. 62 MB vs
   327 MB for this 24-min video.
3. Probe retries (2 attempts) added after a transient Rutube probe
   failure was observed live.
4. --force now wipes artifacts first (yt-dlp silently skips an existing
   audio.wav otherwise - found while validating the force path).
5. Intermediates removed after verify; a workdir ends with audio.wav only.

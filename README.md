# CoreCast

Pipeline: public web video (YouTube / VK / Rutube) -> structured summary text.

1. Downloads only the audio track of a video given its URL (yt-dlp).
2. Transcribes it locally on GPU (NeMo-Speech.cpp, Parakeet-TDT-0.6B-v3).
3. Summarizes the transcript with the DeepSeek API (deepseek-v4-pro, deep
   thinking): numbered sections with short titles, in the language of the
   video, ending with a closing summary thought.

## Status

Stage-based development, each stage is discussed and approved before
implementation.

- [x] Stage 1 (URL -> audio WAV)
- [x] Stage 2 (audio -> transcript)
- [x] Stage 3 (transcript -> summary)
- [x] Stage 4 (pywebview GUI)
- [ ] Stage 5 (packaging): PyInstaller exe (deferred)

## Deployment from scratch (Windows)

### 1. Prerequisites

- Windows 10/11, NVIDIA GPU with >= 8 GB VRAM recommended
  (verified on an RTX 4060 8 GB; peak usage ~5 GB)
- Python 3.11, git, and **ffmpeg + ffprobe on PATH**
- For building the ASR engine only: Visual Studio with the "Desktop
  development with C++" workload, CMake >= 3.26, Ninja, and the CUDA
  toolkit (12.x/13.x, `nvcc` on PATH)
- DeepSeek API key: https://platform.deepseek.com (paid; the summarize
  stage needs it)

### 2. Get the code and the Python environment

```powershell
git clone https://github.com/avgorshk/corecast.git
cd corecast
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `yt-dlp`, `certifi` (required: Windows venv Python
has no CA store and VK's CDN fails SSL verification without it), and
`pywebview` (GUI).

### 3. Build the ASR engine (project-local)

The transcription engine is NVIDIA's NeMo-Speech.cpp, built from source
into the project (nothing is installed globally):

```powershell
git clone https://github.com/NVIDIA/NeMo-Speech.cpp.git vendor\NeMo-Speech.cpp
cd vendor\NeMo-Speech.cpp
git submodule update --init ggml
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 `
    -Backend cuda -Config Release -AsrOnly -BuildDir build-cuda-asr
cd ..\..
# the build tree bin/ is self-contained (DLLs next to the exe): copy flat
xcopy /E /I vendor\NeMo-Speech.cpp\build-cuda-asr\bin bin
```

Known build issues:

- **VS 2026/Insiders**: `build.ps1` fails with "No VS install with the C++
  toolset for x64 found" because its vswhere call lacks `-prerelease` -
  add it (around line 241 of the script).
- **CUDA 13.x**: the full build dies in the TTS core
  (`magpietts_cuda_sampling.cu`, MSVC preprocessor). `-AsrOnly` skips TTS
  and builds the ASR targets cleanly; if an ASR file ever hits the same
  error, add `-DCMAKE_CUDA_FLAGS=/Zc:preprocessor`.
- At run time the CUDA toolkit `bin` must be on PATH (or pass
  `-CublasShim` to bundle cuBLAS).

Check the engine: `bin\nemo-speech.exe doctor` should detect the GPU.

### 4. Download the ASR model

```powershell
curl -L https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.q8_0.gguf `
     -o models\parakeet-tdt-0.6b-v3.q8_0.gguf
```

~681 MB. Verify the sha256 against the CDN's `X-Linked-ETag` header:
`curl -sIL <same-URL>` (the ETag is the file's sha256).

### 5. Configure the DeepSeek key

Create `.env` in the repo root (gitignored):

```
DEEPSEEK_API_KEY=sk-...
```

### 6. Run

CLI, stage by stage:

```
python fetch.py <url> [--out-dir DIR] [--force] [-v] [--json-progress]
python transcribe.py <audio.wav> [--model PATH] [--device cuda] [--format text|json] [--force]
python summarize.py <transcript.txt> [--model M] [--effort low|high|max] [--force]
```

GUI (double-clickable icon, no console):

```powershell
powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
# double-click CoreCast.lnk
# or just: python gui.py
```

The GUI wraps the same three stages via `pipeline.py`; run artifacts go
to `%LOCALAPPDATA%\CoreCast\run\` and are wiped when the app window
closes.

## Usage notes

- `fetch.py` - URL -> 16 kHz mono WAV. Handles the platform quirks:
  audio-only streams where they exist (YouTube, VK), and on Rutube
  (combined-only) it sample-probes the smallest renditions and picks the
  cheapest one with adequate audio - it never downloads a full video.
- `transcribe.py` - WAV -> transcript (Parakeet-TDT-0.6B-v3, CUDA,
  ~50x realtime on an RTX 4060; 24 min of audio in ~30 s).
- `summarize.py` - transcript -> structured summary. Deep thinking on by
  default (`thinking` enabled, `reasoning_effort=max`; tune with
  `--effort low|high|max`). Format: at most 5 numbered sections with
  short titles + closing summary thought, whole answer <= 4000
  characters. `--model deepseek-v4-flash`: ~3x faster and
  much cheaper, leaner output. Thinking tokens are billed as output
  tokens.

## Troubleshooting

- **VK: "unable to download video data: [SSL: CERTIFICATE_VERIFY_FAILED]"**
  -> `pip install -r requirements.txt` (certifi) is missing.
- **VK: "no audio-only and no combined formats available"** -> transient
  empty format list; retry. Repeats may mean the video needs login.
- **VK/Rutube downloads crawl** -> throttled CDNs; the picker already
  selects the lowest adequate tier, patience is normal there.
- **GUI opens with a demo animation** -> you are running the old build or
  a plain browser; the real app shows "Engine connected" in the status
  bar on start.
- **Console windows flash during a GUI run** -> fixed via
  `CREATE_NO_WINDOW` at every child spawn; if a new one appears, report
  the stage.

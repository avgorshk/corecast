#!/usr/bin/env python3
"""CoreCast Stage 4: pipeline runner for the GUI.

Runs the three CLI stages sequentially as subprocesses, parses their
--json-progress event streams, and maintains a thread-safe state dict
the GUI polls.

Progress mapping (internal stage weights, invisible to the user):
    download 0..50   fetch.py: download phase bytes% -> 0..45, convert -> 50
    transcribe 50..90  transcribe.py phase percent, as-is
    summarize 90..100  token stream vs estimated output size, capped at 99
                       until the stream ends

CLI smoke test:  python pipeline.py <url>
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.stdout is None:          # running under pythonw.exe
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(os.environ.get("LOCALAPPDATA", str(PROJECT_ROOT))) / "CoreCast" / "run"

STAGES = [
    ("download",   "fetch.py",      0.0, 0.50),
    ("transcribe", "transcribe.py", 0.50, 0.90),
    ("summarize",  "summarize.py",  0.90, 1.00),
]

IDLE = {
    "status": "idle", "stage": "idle", "total_pct": 0,
    "message": "Ready", "error": None, "summary": None, "wall_s": 0.0,
    "output_dir": None,
}


class PipelineRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = dict(IDLE)
        self._busy = False

    # ---------------------------------------------------------- public API

    def start(self, url):
        """Validate URL and spawn the worker. Returns None or an error."""
        url = (url or "").strip()
        if not url:
            return "error: paste a video URL first"
        if not re.match(r"^https?://\S+$", url, re.IGNORECASE):
            return "error: not a valid video URL"
        with self._lock:
            if self._busy:
                return "error: already running"
            self._busy = True
            self._state = dict(IDLE, status="running", stage="download",
                               message="Download: starting ...")
        threading.Thread(target=self._run, args=(url,), daemon=True).start()
        return None

    def state(self):
        with self._lock:
            return dict(self._state)

    # ------------------------------------------------------------ pipeline

    def _run(self, url):
        try:
            run_dir = RUN_ROOT / self._run_id(url)
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            # fetch.py nests the workdir under <out-dir>/<video-id>/;
            # transcribe/summarize derive their outputs next to the audio
            audio = None
            transcript = None
            summary = None

            # stage 1: fetch
            self._spawn("fetch.py", [url, "--out-dir", str(run_dir)],
                        self._on_fetch)
            # fetch.py nests the workdir under <out-dir>/<video-id>/
            audio = next(run_dir.rglob("audio.wav"), None)
            if audio is None:
                raise PipelineError("fetch.py completed but audio.wav "
                                    "was not produced")
            transcript = audio.with_name("transcript.txt")
            summary = audio.with_name("summary.txt")
            # stage 2: transcribe
            self._spawn("transcribe.py", [str(audio)],
                        self._on_transcribe)
            # stage 3: summarize
            est = self._estimate_tokens(transcript)
            self._spawn("summarize.py", [str(transcript)],
                        lambda ev, txt: self._on_summarize(ev, txt, est))

            text = summary.read_text(encoding="utf-8", errors="replace")
            with self._lock:
                self._state.update(status="done", stage="idle",
                                   total_pct=100, message="Done",
                                   summary=text.strip(),
                                   wall_s=time.perf_counter() - t0,
                                   output_dir=str(run_dir))
        except PipelineError as e:
            with self._lock:
                self._state.update(status="error", error=str(e),
                                   message="Error")
        except Exception as e:  # unexpected: report, don't hang the GUI
            with self._lock:
                self._state.update(status="error", error=f"internal: {e}",
                                   message="Error")
        finally:
            with self._lock:
                self._busy = False

    def _spawn(self, script, args, on_event):
        """Run one stage script with --json-progress, feeding events."""
        cmd = [sys.executable, str(PROJECT_ROOT / script),
               "--json-progress"] + args
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace",
                                bufsize=1, creationflags=creationflags)
        err_tail = ""
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # plain-text output (pre-reporter errors, tool warnings)
                if line.startswith("ERROR"):
                    raise PipelineError(line)
                if line.startswith(("OK", "SKIP")):
                    on_event({}, line)
                continue
            if ev.get("type") == "phase":
                on_event(ev, "")
            elif ev.get("type") == "status":
                txt = ev.get("text", "")
                if txt.startswith("ERROR"):
                    raise PipelineError(txt)
                on_event(ev, txt)
        for line in proc.stderr:
            err_tail = (err_tail + line)[-300:]
        proc.wait()
        if proc.returncode != 0:
            raise PipelineError(
                f"{script} failed (exit {proc.returncode}): {err_tail.strip()}")

    # ------------------------------------------------------- progress map

    def _on_fetch(self, ev, txt):
        if ev.get("phase") == "download":
            pct = ev.get("percent")
            if pct is not None:
                self._update("download", int(pct * 0.45),
                             f"Download: downloading audio {int(pct)}% ...")
            elif ev.get("status") == "done":
                self._update("download", 45, "Download: converting to WAV ...")
        elif txt.startswith("OK"):
            self._update("download", 50, "Download: done")

    def _on_transcribe(self, ev, txt):
        if ev.get("phase") == "transcribe":
            pct = ev.get("percent")
            if pct is not None:
                self._update("transcribe", 50 + int(pct * 0.4),
                             f"Transcribe: transcribing on GPU {int(pct)}% "
                             "(~48x realtime)")
        elif txt.startswith("OK"):
            self._update("transcribe", 90, "Transcribe: done")

    def _on_summarize(self, ev, txt, est):
        if ev.get("phase") in ("summarize", "merge"):
            tok = ev.get("done") or 0
            frac = min(tok / est, 0.99)
            self._update("summarize", 90 + int(10 * frac),
                         f"Summarize: {int(tok)} tokens "
                         f"(deepseek-v4-flash)")
        elif txt.startswith("OK"):
            self._update("summarize", 100, "Summarize: done")

    def _estimate_tokens(self, transcript):
        """Output-length estimate for the indeterminate token stream."""
        chars = transcript.stat().st_size
        return max(400, min(int(chars / 2.2 * 0.10), 2500))

    def _update(self, stage, total_pct, message):
        with self._lock:
            if self._state.get("status") == "running":
                self._state.update(stage=stage, total_pct=total_pct,
                                   message=message)

    @staticmethod
    def _run_id(url):
        slug = re.sub(r"[^A-Za-z0-9]+", "-", url)[:30].strip("-")
        return f"{slug or 'video'}_{time.strftime('%Y%m%d-%H%M%S')}"


class PipelineError(Exception):
    pass


if __name__ == "__main__":          # CLI smoke test
    r = PipelineRunner()
    err = r.start(sys.argv[1])
    if err:
        print(err); sys.exit(1)
    last = None
    while True:
        st = r.state()
        if st != last:
            print(f"{st['status']:8s} {st['total_pct']:3d}%  "
                  f"{st['stage']:11s} {st['message']}")
            last = st
        if st["status"] in ("done", "error"):
            if st["error"]:
                print("ERROR:", st["error"])
            elif st["summary"]:
                print(f"\n{st['summary'][:600]}")
            sys.exit(0 if st["status"] == "done" else 1)
        time.sleep(0.1)

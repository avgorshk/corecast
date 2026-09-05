#!/usr/bin/env python3
"""CoreCast Stage 1: download the audio track of a public web video.

Platforms: YouTube / VK / Rutube (via yt-dlp, in-process).

Usage: python fetch.py <url> [--out-dir DIR] [--force] [-v] [--json-progress]

Output: <out-dir>/<video_id>/audio.wav  (16 kHz mono PCM s16le)

Format selection:
  - audio-only stream available (YouTube, VK): take bestaudio.
  - only combined video+audio streams exist (Rutube): sample-probe the
    smallest renditions with ffmpeg and pick the first whose audio is
    >= 64 kbps (audio-only renditions: >= 48 kbps), so we never download
    a full-size video just for its audio.

Progress reporting:
  - interactive console: ASCII progress bar for the download phase.
  - piped/captured output: periodic progress lines (~10% or 30 s).
  - --json-progress: one JSON event per line, for CI logs or a future GUI.

Exit codes: 0 ok/skip | 1 yt-dlp failure | 2 live stream | 3 tools missing |
            4 verification failed | 5 usage error
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "data"

AUDIO_MIN_KBPS = 64.0     # probe threshold: speech-quality floor (~64 kbps AAC
                         # is transparent for voice). Higher tiers on throttled
                         # platforms (rutube ~0.3 MB/s) cost 5x wall time.
AUDIO_ONLY_MIN_KBPS = 48.0  # lower floor for TRUE audio-only renditions:
                            # 53 kbps audio-only beats 69 kbps audio buried in
                            # a 600+ MB combined file (download economics).
MAX_PROBES = 4            # how many renditions to sample at most
PROBE_SECONDS = 12        # sample length per rendition
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")

BAR_W = 24                # width of the ASCII progress bar


# ---------------------------------------------------------------- reporters

class Reporter:
    """Progress sink. Fetch logic emits events; renderers decide how to show
    them (CLI bar now, GUI widget later, JSON stream for machine consumers)."""

    def status(self, text):
        raise NotImplementedError

    def phase_start(self, phase, label):
        pass

    def phase_update(self, phase, done, total=None, extra=None):
        pass

    def phase_done(self, phase):
        pass


def human_bytes(n):
    if n < 1024:
        return f"{int(n)} B"
    n /= 1024
    if n < 1024:
        return f"{n:.1f} KiB"
    n /= 1024
    if n < 1024:
        return f"{n:.1f} MiB"
    return f"{n / 1024:.2f} GiB"


def human_speed(bps):
    if bps is None:
        return ""
    return human_bytes(bps) + "/s"


def fmt_eta(sec):
    if sec is None:
        return "?"
    s = max(0, int(sec))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 \
        else f"{s // 60}:{s % 60:02d}"


def compose_bar(label, done, total, unit, speed, eta):
    if total:
        pct = min(100.0, done / total * 100)
        filled = int(round(BAR_W * pct / 100))
        bar = "[" + "#" * filled + "-" * (BAR_W - filled) + f"] {pct:3.0f}%"
    else:
        bar = "[  downloading ...  ]"
    if unit == "B":
        size = human_bytes(done)
        if total:
            size += "/" + human_bytes(total)
    elif unit == "s":
        size = f"{done:.1f}s/{total:.1f}s"
    else:
        size = f"{done}/{total} fragments"
    bits = [bar, size]
    if speed:
        bits.append(human_speed(speed))
    if eta is not None and total:
        bits.append("ETA " + fmt_eta(eta))
    return label + "  ".join(bits)


class CLIReporter(Reporter):
    """ASCII bar on a TTY; periodic lines when output is captured."""

    def __init__(self, tty):
        self.tty = tty
        self._bar = None
        self._mid_line = False
        self._last_pct = None
        self._last_t = 0.0

    def status(self, text):
        self._end_bar()
        print(text, flush=True)

    def phase_start(self, phase, label):
        self._bar = {"phase": phase, "label": label,
                     "done": 0, "total": None, "extra": {}}

    def phase_update(self, phase, done, total=None, extra=None):
        if self._bar is None or self._bar["phase"] != phase:
            return
        self._bar["done"] = done
        self._bar["total"] = total
        self._bar["extra"] = extra or {}
        self._render()

    def phase_done(self, phase):
        if self._bar is not None and self._bar["phase"] == phase:
            self._end_bar()
            self._bar = None

    def _render(self):
        b = self._bar
        ex = b["extra"]
        line = compose_bar(b["label"], b["done"], b["total"],
                           ex.get("unit", "B"), ex.get("speed"), ex.get("eta"))
        if self.tty:
            sys.stdout.write("\r" + line.ljust(90))
            sys.stdout.flush()
            self._mid_line = True
            return
        total = b["total"]
        pct = b["done"] / total * 100 if total else None
        now = time.perf_counter()
        if total and pct >= 100:  # final tick: always print the 100% line
            if self._last_pct != 100:
                print(line, flush=True)
                self._last_pct = 100.0
                self._last_t = now
            return
        due = (pct is not None and self._last_pct is not None
               and pct - self._last_pct >= 10) or (now - self._last_t >= 30)
        if due or self._last_pct is None:
            print(line, flush=True)
            self._last_pct = pct
            self._last_t = now

    def _end_bar(self):
        if self._mid_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._mid_line = False


class JSONReporter(Reporter):
    """One JSON object per line: parseable by CI or a GUI wrapper process."""

    def _emit(self, obj):
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    def status(self, text):
        self._emit({"type": "status", "text": text})

    def phase_start(self, phase, label):
        self._emit({"type": "phase", "phase": phase, "status": "start",
                    "label": label})

    def phase_update(self, phase, done, total=None, extra=None):
        ex = extra or {}
        obj = {"type": "phase", "phase": phase, "status": "update",
               "done": done, "total": total}
        if total:
            obj["percent"] = round(min(100.0, done / total * 100), 2)
        if ex.get("speed") is not None:
            obj["speed_bps"] = ex["speed"]
        if ex.get("eta") is not None:
            obj["eta_s"] = ex["eta"]
        if ex.get("unit"):
            obj["unit"] = ex["unit"]
        self._emit(obj)

    def phase_done(self, phase):
        self._emit({"type": "phase", "phase": phase, "status": "done"})


# ------------------------------------------------------------------ utils

def find_tool(name):
    return shutil.which(name)


def human_size(nbytes):
    return f"{nbytes / (1024 * 1024):.1f} MiB"


def fmt_dur(sec):
    if sec is None:
        return "?"
    s = int(round(sec))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def silent_logger():
    """A logger that drops everything: yt-dlp must not print its own error
    lines - we format and report errors ourselves (single clean message)."""
    lg = logging.getLogger("corecast.ytdlp")
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())
    lg.propagate = False
    lg.setLevel(logging.CRITICAL)
    return lg


def fetch_meta(url, attempts=2):
    """In-process yt-dlp metadata with retry. Also retries when the platform
    returns an empty format list (observed transiently on VK)."""
    last_err = "unknown yt-dlp error"
    for _ in range(attempts):
        ydl = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                "noplaylist": True, "socket_timeout": 30,
                                "logger": silent_logger()})
        try:
            meta = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            last_err = str(e).removeprefix("ERROR: ")
            time.sleep(2)
            continue
        if meta and meta.get("formats"):
            return meta, None
        last_err = ("platform returned no playable formats "
                    "(transient or login required; try again later)")
        time.sleep(2)
    return None, last_err


# ----------------------------------------------------------------- checks

def check_availability(meta):
    """Refuse live streams and age-restricted videos. Return (ok, reason)."""
    if meta.get("is_live"):
        return False, "live stream detected; only recorded videos are supported"
    age = meta.get("age_limit") or 0
    if age > 0:
        return False, f"age-restricted video (age_limit={age})"
    return True, ""


def probe_audio(path):
    """ffprobe -> dict(codec, sample_rate, channels, duration_s) or None."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        return None
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format",
         "-print_format", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30)
    if r.returncode != 0:
        return None
    try:
        p = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    for st in p.get("streams", []):
        if st.get("codec_type") == "audio":
            dur = None
            try:
                dur = float(p["format"]["duration"])
            except (KeyError, ValueError, TypeError):
                pass
            return {"codec": st.get("codec_name"),
                    "sample_rate": st.get("sample_rate"),
                    "channels": st.get("channels"),
                    "duration_s": dur}
    return None


def verify(audio_path, meta_duration):
    """Return (passed, check_labels, details)."""
    info = probe_audio(audio_path)
    if info is None:
        return False, ["ffprobe could not read the file"], {}
    checks = [
        ("codec", info["codec"] == "pcm_s16le"),
        ("sample_rate", info["sample_rate"] == "16000"),
        ("channels", info["channels"] == 1),
    ]
    if meta_duration is not None and info["duration_s"] is not None:
        checks.append(("duration", abs(info["duration_s"] - meta_duration) <= 1.0))
    labels = "/".join(name for name, _ in checks)
    passed = all(ok for _, ok in checks)
    return passed, [f"{'PASS' if passed else 'FAIL'}: {labels}"], info


def best_audio_format(meta):
    """Human-readable description of the bestaudio format from the metadata."""
    cands = [f for f in meta.get("formats", []) or []
             if f.get("vcodec") in ("none", None) and f.get("acodec") != "none"]
    if not cands:
        return "?"
    best = max(cands, key=lambda f: f.get("abr") or 0)
    if not best.get("abr"):
        best = max(cands, key=lambda f: f.get("tbr") or 0)
    ext = best.get("ext", "?")
    abr = best.get("abr")
    note = best.get("format_note") or ""
    desc = ext if not abr else f"{ext} {abr:.0f} kbps"
    return f"{desc} ({note})" if note else desc


# ---------------------------------------------------------- format picking

def sample_audio_kbps(url, tmpdir, attempts=2):
    """Download a short audio sample of a rendition and measure its bitrate.

    Retries once on transient failure (HTTP hiccups on throttled CDNs).
    Sends a browser User-Agent (VK 403s ffmpeg's default UA).
    """
    out = Path(tmpdir) / "probe.mka"
    for _ in range(attempts):
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-user_agent", USER_AGENT,
                 "-i", url,
                 "-t", str(PROBE_SECONDS), "-map", "0:a:0", "-c", "copy",
                 str(out)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60)
        except subprocess.TimeoutExpired:
            r = None
        if r is not None and r.returncode == 0 and out.is_file():
            pr = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(out)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=20)
            try:
                dur = float(pr.stdout.strip())
                if dur > 0:
                    return out.stat().st_size * 8 / dur / 1000
            except ValueError:
                pass
        time.sleep(1.5)
    return None


def select_format(meta, rep, verbose=False):
    """Return (format_selector, reason).

    - audio-only formats exist: "bestaudio"
    - otherwise: sample the smallest candidates, pick the first whose audio
      reaches its floor (48 kbps for audio-only ids, 64 kbps otherwise)
    """
    fmts = [f for f in meta.get("formats", []) or [] if f.get("url")]
    audio_only = [f for f in fmts
                  if f.get("vcodec") in (None, "none")
                  and f.get("acodec") not in (None, "none")]
    if audio_only:
        return "bestaudio", f"audio-only: {best_audio_format(meta)}"

    # candidates: anything not clearly video-only. Some platforms (VK,
    # intermittently) report formats with UNKNOWN codecs - the probe decides.
    def clearly_video_only(f):
        return (f.get("vcodec") not in (None, "none")
                and f.get("acodec") in (None, "none"))

    combined = [f for f in fmts if not clearly_video_only(f)]
    if not combined:
        return None, ("no downloadable formats (all entries are video-only "
                      "or unsupported)")

    combined.sort(key=lambda f: f.get("tbr") or (f.get("height") or 0) * 10)
    rep.status(f"[2/4] audio     -> no audio-only streams; probing renditions "
               f"(up to {MAX_PROBES}) ...")
    tmpdir = tempfile.mkdtemp(prefix="corecast_probe_")
    best = None
    probed = 0
    try:
        for f in combined[:MAX_PROBES]:
            probed += 1
            kbps = sample_audio_kbps(f["url"], tmpdir)
            desc = f"{f.get('height')}p" if f.get("height") else str(f.get("format_id"))
            rep.status(f"                   candidate {probed}/{MAX_PROBES}: "
                       f"{desc} -> {f'{kbps:.0f} kbps' if kbps is not None else 'probe failed'}")
            if kbps is not None:
                if best is None or kbps > best[0]:
                    best = (kbps, f)
                fid = str(f.get("format_id") or "").lower()
                thr = AUDIO_ONLY_MIN_KBPS if "audio" in fid else AUDIO_MIN_KBPS
                if kbps >= thr:
                    break
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if best is None:
        f = combined[0]
        return f.get("format_id"), (f"audio-only: none; probes failed; "
                                    f"using smallest candidate ({f.get('format_id')})")
    kbps, f = best
    desc = f"{f.get('height')}p" if f.get("height") else str(f.get("format_id"))
    return f.get("format_id"), (f"audio-only: none; probed {probed} candidate(s); "
                                f"picked {desc} with ~{kbps:.0f} kbps audio")


# -------------------------------------------------------------- download

def make_hook(rep):
    def hook(d):
        st = d.get("status")
        if st == "downloading":
            fc = d.get("fragment_count")
            if fc:
                rep.phase_update("download", d.get("fragment_index") or 0, fc,
                                 {"unit": "frag", "speed": d.get("speed"),
                                  "eta": d.get("eta")})
            else:
                rep.phase_update("download", d.get("downloaded_bytes") or 0,
                                 d.get("total_bytes") or d.get("total_bytes_estimate"),
                                 {"unit": "B", "speed": d.get("speed"),
                                  "eta": d.get("eta")})
        elif st == "finished":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or total or 0
            rep.phase_update("download", done, total,
                             {"unit": "B", "speed": 0, "eta": 0})
    return hook


def download(url, sel, workdir, rep):
    """In-process yt-dlp download of the selected format into workdir.

    Returns an error string, or None on success.
    """
    params = {"quiet": True, "no_warnings": True, "noprogress": True,
              "noplaylist": True, "socket_timeout": 30,
              "format": sel,
              "outtmpl": "audio.%(ext)s",
              "paths": {"home": str(workdir)},
              "overwrites": True,
              "logger": silent_logger(),
              "progress_hooks": [make_hook(rep)]}
    ydl = yt_dlp.YoutubeDL(params)
    rep.phase_start("download", "[3/4] download   ")
    try:
        ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        rep.phase_done("download")
        return str(e).removeprefix("ERROR: ")
    rep.phase_done("download")
    srcs = [p for p in workdir.glob("audio.*") if p.suffix.lower() != ".wav"]
    return None if srcs else "downloaded file not found after download"


def convert_to_wav(src, dst):
    """ffmpeg: any container -> 16 kHz mono PCM WAV. Returns error or None."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=900)
    except subprocess.TimeoutExpired:
        return "ffmpeg conversion timed out"
    if r.returncode != 0 or not dst.is_file():
        tail = (r.stderr.strip().splitlines() or ["conversion failed"])[-1]
        return f"ffmpeg conversion failed: {tail}"
    return None


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="fetch.py",
                                 description="CoreCast Stage 1: URL -> 16 kHz mono WAV")
    ap.add_argument("url")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                    help="re-download even if audio.wav exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json-progress", action="store_true",
                    help="machine-readable progress events (one JSON line each)")
    args = ap.parse_args(argv)

    # preflight
    if yt_dlp is None:
        print("ERROR: yt-dlp is not installed (pip install -r requirements.txt) (exit 3)")
        return 3
    if not find_tool("ffmpeg") or not find_tool("ffprobe"):
        print("ERROR: ffmpeg/ffprobe not found on PATH (exit 3)")
        return 3
    if "://" not in args.url:
        print("ERROR: not a URL (exit 5)")
        return 5

    rep = JSONReporter() if args.json_progress else CLIReporter(sys.stdout.isatty())
    t0 = time.perf_counter()

    # [1/4] metadata (in-memory only; nothing is persisted)
    meta, err = fetch_meta(args.url)
    if meta is None:
        rep.status(f"ERROR: {err} (exit 1)")
        return 1

    ok, why = check_availability(meta)
    if not ok:
        code = 2 if meta.get("is_live") else 1
        rep.status(f"ERROR: {why} (exit {code})")
        return code

    vid = meta.get("id") or Path(args.url.rstrip("/")).name
    duration = meta.get("duration")
    rep.status(f"[1/4] info      -> duration={fmt_dur(duration)}, "
               f"live=no, platform={meta.get('extractor', '?')}")

    workdir = Path(args.out_dir) / vid
    workdir.mkdir(parents=True, exist_ok=True)
    audio = workdir / "audio.wav"

    # --force: wipe all artifacts so yt-dlp truly re-downloads
    if args.force:
        for leftover in workdir.glob("audio.*"):
            leftover.unlink(missing_ok=True)

    # existing verified file -> skip
    if audio.is_file() and not args.force:
        passed, lines, info = verify(audio, duration)
        if passed:
            rep.status(f"[4/4] verify    -> [{lines[0]}] (existing file, skipped download)")
            rep.status(f"SKIP {workdir} | {fmt_dur(duration)} | "
                       f"{human_size(audio.stat().st_size)}")
            return 0
        rep.status(f"[4/4] verify    -> [{lines[0]}] existing file corrupt, re-downloading")
        audio.unlink()

    # [2/4] format selection: audio-only preferred, probe-based fallback
    sel, reason = select_format(meta, rep, args.verbose)
    if sel is None:
        rep.status(f"ERROR: {reason} (exit 1)")
        return 1
    rep.status(f"[2/4] audio     -> {reason}")

    # [3/4] download (with progress) + convert
    t1 = time.perf_counter()
    err = download(args.url, sel, workdir, rep)
    if err:
        rep.status(f"ERROR: {err} (exit 1)")
        return 1
    dt = time.perf_counter() - t1
    rep.status(f"[3/4] download   done ({dt:.1f}s)")

    srcs = [p for p in workdir.glob("audio.*") if p.suffix.lower() != ".wav"]
    src = srcs[0]
    t2 = time.perf_counter()
    err = convert_to_wav(src, audio)
    if err:
        rep.status(f"ERROR: {err} (exit 1)")
        return 1
    dtc = time.perf_counter() - t2
    size = audio.stat().st_size
    rep.status(f"[3/4] convert   -> audio.wav, {human_size(size)}, {dtc:.1f}s")
    src.unlink(missing_ok=True)

    # [4/4] verify
    passed, lines, info = verify(audio, duration)
    rep.status(f"[4/4] verify    -> [{lines[0]}]")
    if not passed:
        return 4

    wall = time.perf_counter() - t0
    dur = info.get("duration_s") or duration
    rep.status(f"OK  {workdir} | {fmt_dur(dur)} | {human_size(size)} | {wall:.1f}s wall")
    return 0


if __name__ == "__main__":
    sys.exit(main())

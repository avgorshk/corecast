#!/usr/bin/env python3
"""CoreCast Stage 1: download the audio track of a public web video.

Platforms: YouTube / VK / Rutube (via yt-dlp).

Usage: python fetch.py <url> [--out-dir DIR] [--force] [-v]

Output: <out-dir>/<video_id>/audio.wav  (16 kHz mono PCM s16le)

Format selection:
  - audio-only stream available (YouTube, VK): take bestaudio.
  - only combined video+audio streams exist (Rutube): sample-probe the
    smallest renditions with ffmpeg and pick the first whose audio is
    >= 96 kbps, so we never download a full-size video just for its audio.

Exit codes: 0 ok/skip | 1 yt-dlp failure | 2 live stream | 3 tools missing |
            4 verification failed | 5 usage error
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


# ------------------------------------------------------------------ utils

def find_ytdlp():
    exe = shutil.which("yt-dlp")
    if exe:
        return exe
    for name in ("yt-dlp.exe", "yt-dlp"):  # venv layout: next to python.exe
        cand = Path(sys.executable).with_name(name)
        if cand.is_file():
            return str(cand)
    return None


def find_tool(name):
    return shutil.which(name)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def human_size(nbytes):
    return f"{nbytes / (1024 * 1024):.1f} MiB"


def fmt_dur(sec):
    if sec is None:
        return "?"
    s = int(round(sec))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fetch_meta(ytdlp, url, attempts=2):
    """yt-dlp -J with retry. Also retries when the platform returns an empty
    format list (observed transiently on VK, likely IP rate-limiting)."""
    last_err = "unknown yt-dlp error"
    for _ in range(attempts):
        try:
            r = run([ytdlp, "-J", "--no-playlist", "--no-warnings", url],
                    timeout=90)
        except subprocess.TimeoutExpired:
            last_err = "metadata request timed out"
            time.sleep(2)
            continue
        if r.returncode != 0:
            last_err = (r.stderr.strip().splitlines() or [last_err])[-1]
            last_err = last_err.removeprefix("ERROR: ")
            time.sleep(2)
            continue
        try:
            meta = json.loads(r.stdout)
        except json.JSONDecodeError:
            last_err = "could not parse yt-dlp output"
            time.sleep(2)
            continue
        if meta.get("formats"):
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
    r = run([ffprobe, "-v", "error", "-show_streams", "-show_format",
             "-print_format", "json", str(path)])
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
    """Human-readable description of the bestaudio format from -J dump."""
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
    """
    out = Path(tmpdir) / "probe.mka"
    for _ in range(attempts):
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-user_agent", USER_AGENT,
             "-i", url,
             "-t", str(PROBE_SECONDS), "-map", "0:a:0", "-c", "copy", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60)
        if r.returncode == 0 and out.is_file():
            pr = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(out)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=20)
            try:
                dur = float(pr.stdout.strip())
                if dur > 0:
                    return out.stat().st_size * 8 / dur / 1000
            except ValueError:
                pass
        time.sleep(1.5)
    return None


def select_format(meta, verbose=False):
    """Return (format_selector, reason).

    - audio-only formats exist: "bestaudio"
    - combined-only platform: sample the smallest renditions, pick the
      first with audio >= AUDIO_MIN_KBPS (or the best sampled as fallback)
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
    print(f"[2/4] audio     -> no audio-only streams; probing renditions "
          f"(up to {MAX_PROBES}) ...")
    tmpdir = tempfile.mkdtemp(prefix="corecast_probe_")
    best = None
    probed = 0
    try:
        for f in combined[:MAX_PROBES]:
            probed += 1
            kbps = sample_audio_kbps(f["url"], tmpdir)
            if verbose:
                print(f"                   {f.get('format_id')} "
                      f"({f.get('height')}p): {kbps and f'{kbps:.0f} kbps' or 'probe failed'}")
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


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="fetch.py",
                                 description="CoreCast Stage 1: URL -> 16 kHz mono WAV")
    ap.add_argument("url")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                    help="re-download even if audio.wav exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    # preflight
    ytdlp = find_ytdlp()
    if not ytdlp:
        print("ERROR: yt-dlp not found (pip install yt-dlp) (exit 3)")
        return 3
    if not find_tool("ffmpeg") or not find_tool("ffprobe"):
        print("ERROR: ffmpeg/ffprobe not found on PATH (exit 3)")
        return 3
    if "://" not in args.url:
        print("ERROR: not a URL (exit 5)")
        return 5

    t0 = time.perf_counter()

    # [1/4] metadata (in-memory only; nothing is persisted)
    meta, err = fetch_meta(ytdlp, args.url)
    if meta is None:
        print(f"ERROR: {err} (exit 1)")
        return 1

    ok, why = check_availability(meta)
    if not ok:
        code = 2 if meta.get("is_live") else 1
        print(f"ERROR: {why} (exit {code})")
        return code

    vid = meta.get("id") or Path(args.url.rstrip("/")).name
    duration = meta.get("duration")
    print(f"[1/4] info      -> duration={fmt_dur(duration)}, "
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
            print(f"[4/4] verify    -> [{lines[0]}] (existing file, skipped download)")
            print(f"SKIP {workdir} | {fmt_dur(duration)} | "
                  f"{human_size(audio.stat().st_size)}")
            return 0
        print(f"[4/4] verify    -> [{lines[0]}] existing file corrupt, re-downloading")
        audio.unlink()

    # format selection: audio-only preferred, probe-based fallback
    sel, reason = select_format(meta, args.verbose)
    if sel is None:
        print(f"ERROR: {reason} (exit 1)")
        return 1
    print(f"[2/4] audio     -> {reason}")

    # [2/4] + [3/4] one yt-dlp run: download, convert to WAV
    t1 = time.perf_counter()
    cmd = [ytdlp,
           "-f", sel,
           "-x", "--audio-format", "wav",
           "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
           "-o", "audio.%(ext)s",
           "--no-playlist", "--no-warnings"]
    if not args.verbose:
        cmd += ["-q"]
    cmd += [args.url]
    try:
        r = run(cmd, cwd=str(workdir), timeout=1800)
    except subprocess.TimeoutExpired:
        print("ERROR: download timed out after 30 min (exit 1)")
        return 1
    if r.returncode != 0 or not audio.is_file():
        tail = (r.stderr.strip().splitlines() or ["download failed"])[-1]
        tail = tail.removeprefix("ERROR: ")
        print(f"ERROR: {tail} (exit 1)")
        return 1
    dt = time.perf_counter() - t1
    size = audio.stat().st_size
    print(f"[3/4] convert   -> audio.wav, {human_size(size)}, {dt:.1f}s")

    # [4/4] verify
    passed, lines, info = verify(audio, duration)
    print(f"[4/4] verify    -> [{lines[0]}]")
    if not passed:
        return 4

    # keep the workdir clean: only audio.wav should remain
    for leftover in workdir.glob("audio.*"):
        if leftover.name != "audio.wav":
            leftover.unlink(missing_ok=True)

    wall = time.perf_counter() - t0
    dur = info.get("duration_s") or duration
    print(f"OK  {workdir} | {fmt_dur(dur)} | {human_size(size)} | {wall:.1f}s wall")
    return 0


if __name__ == "__main__":
    sys.exit(main())

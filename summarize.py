#!/usr/bin/env python3
"""CoreCast Stage 3: transcript -> structured summary via DeepSeek API.

Backend: DeepSeek (paid key). Model: deepseek-v4-flash by default.

Usage: python summarize.py <transcript.txt> [--model M] [--force] [-v]
              [--json-progress]

Key from environment (or the project .env file):
  deepseek: DEEPSEEK_API_KEY (platform.deepseek.com)

Output: <dir>/summary.txt - the major ideas of the transcript, in the
        transcript's language. Author/video mentions ok.

Long transcripts are chunked (map-reduce): each chunk is summarized
separately, the partial summaries are merged into the final text.
DeepSeek's 128K context makes single-pass cover transcripts up to
~200k chars; chunking is a safety net for anything longer.

Exit codes: 0 ok/skip | 1 API failure | 2 input missing |
            3 credentials/config missing | 4 empty output | 5 usage
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fetch import CLIReporter, JSONReporter, USER_AGENT, human_bytes

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"
REASONING_EFFORT = "max"   # thinking effort: low | high | max (API default: high)

SYSTEM_PROMPT = (
    "You are a summarization assistant. Here is a video transcription "
    "text. Please give me the major ideas from it.\n"
    "1. Use the same language as the transcription.\n"
    "2. Use only facts present in the transcript: never invent titles, "
    "numbers, or names.\n"
    "3. You may refer to the author or the video.\n"
    "4. Structure the answer as numbered sections (at most 5) with short "
    "titles, and end with a closing summary thought. Keep the whole "
    "answer within 4000 characters."
)
MAX_OUT_TOKENS = 32768     # thinking + answer share this budget (32k accepted)
CHUNK_CHARS = 200000   # single-pass safety limit; DeepSeek context is 128K tok


# ------------------------------------------------------------------ utils

def load_env():
    """Load KEY=VALUE lines from the project .env into os.environ."""
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def get_config(model, effort):
    """Return (config_dict, error). config: url, model, headers, effort."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None, "DEEPSEEK_API_KEY is not set (platform.deepseek.com)"
    return {"url": DEEPSEEK_URL,
            "model": model or DEEPSEEK_MODEL,
            "effort": effort or REASONING_EFFORT,
            "headers": {"Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "User-Agent": USER_AGENT}}, None


def split_chunks(text, limit):
    """Sentence-level split into chunks of at most `limit` chars."""
    sents = re.split(r"(?<=[.!?\u2026])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > limit and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------- API call

def chat_stream(cfg, messages, rep, phase="summarize", quiet=False):
    """POST chat completions (stream=True), feeding token progress to rep.

    quiet=True: no reporter events (used inside chunked loops).
    Output length is unknowable up front, so the bar is indeterminate:
    a spinner plus a live token counter (total=None).
    Retries on HTTP 429 with a backoff.
    Returns (full_text, error).
    """
    payload = {"model": cfg["model"],
               "messages": messages,
               "thinking": {"type": "enabled"},
               "reasoning_effort": cfg.get("effort", REASONING_EFFORT),
               "max_tokens": MAX_OUT_TOKENS,
               "stream": True}
    data = json.dumps(payload).encode("utf-8")
    if not quiet:
        rep.phase_start(phase, "[2/3] summarize ")

    for attempt in range(5):
        req = urllib.request.Request(cfg["url"], data=data, method="POST",
                                     headers=cfg["headers"])
        text_parts, chars = [], 0
        reasoning_tok = 0
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                    except json.JSONDecodeError:
                        continue
                    if reasoning:
                        reasoning_tok += len(reasoning) // 3
                        if not quiet:
                            # thinking stream: status tick, bar stays put
                            rep.phase_update(phase, reasoning_tok, None,
                                             {"unit": "think"})
                    if content:
                        text_parts.append(content)
                        chars += len(content)
                        if not quiet:
                            # rough token count: indeterminate bar, live
                            # counter (no total - output length unknown)
                            rep.phase_update(phase, chars // 3, None,
                                             {"unit": "tok"})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < 4:
                m = re.search(r"try again in (\d+(?:\.\d+)?)s", body)
                wait = (float(m.group(1)) + 0.5) if m else 15.0
                if not quiet:
                    rep.status(f"                   rate-limited; "
                               f"waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if not quiet:
                rep.phase_done(phase)
            return None, f"API HTTP {e.code}: {body[:300]}"
        except Exception as e:
            if not quiet:
                rep.phase_done(phase)
            return None, f"API error: {e}"

        text = "".join(text_parts).strip()
        if text:
            if not quiet:
                rep.phase_done(phase)
            return text, None
        if attempt < 4:
            time.sleep(3)
            continue
    if not quiet:
        rep.phase_done(phase)
    return None, "API returned empty completion"


# ------------------------------------------------------------ summarize

def summarize_transcript(cfg, rep, transcript):
    """Single-pass when short; chunked map-reduce when long.
    Returns (summary_text, error, n_chunks)."""
    if len(transcript) <= CHUNK_CHARS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Summarize the following transcript.\n\n{transcript}"},
        ]
        text, err = chat_stream(cfg, messages, rep)
        return text, err, 1

    chunks = split_chunks(transcript, CHUNK_CHARS)
    rep.phase_start("summarize", "[2/3] summarize ")
    parts = []
    for i, ch in enumerate(chunks, 1):
        rep.phase_update("summarize", i - 1, len(chunks), {"unit": "chunk"})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": ("Summarize the major ideas from this part of a "
                         "transcript (at most 5 points). Use the same "
                         "language as the transcript. Include specifics."
                         "\n\n" + ch)},
        ]
        text, err = chat_stream(cfg, messages, rep, quiet=True)
        if err:
            rep.phase_done("summarize")
            return None, f"chunk {i}/{len(chunks)}: {err}", len(chunks)
        parts.append(text)
    rep.phase_update("summarize", len(chunks), len(chunks), {"unit": "chunk"})
    rep.phase_done("summarize")

    joined = "\n\n".join(f"--- Part {i + 1} ---\n{p}"
                         for i, p in enumerate(parts))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": ("Combine these partial summaries of the same video "
                     "into ONE final summary of the major ideas: at most "
                     "5 numbered sections with short titles, ending with "
                     "a closing summary thought, the whole answer within "
                     "4000 characters. Remove duplicated points."
                     f"\n\n{joined}")},
    ]
    text, err = chat_stream(cfg, messages, rep, phase="merge")
    return text, err, len(chunks)


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="summarize.py",
                                 description="CoreCast Stage 3: transcript -> summary")
    ap.add_argument("transcript", help="input transcript file")
    ap.add_argument("--model", default=DEEPSEEK_MODEL,
                    help=f"DeepSeek model (default: {DEEPSEEK_MODEL})")
    ap.add_argument("--effort", choices=["low", "high", "max"],
                    default=REASONING_EFFORT,
                    help=f"thinking effort (default: {REASONING_EFFORT})")
    ap.add_argument("--force", action="store_true",
                    help="re-summarize even if summary.txt exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json-progress", action="store_true")
    args = ap.parse_args(argv)

    load_env()
    rep = JSONReporter() if args.json_progress else CLIReporter(sys.stdout.isatty())

    t = Path(args.transcript)
    if not t.is_file():
        rep.status(f"ERROR: transcript not found: {t} (exit 2)")
        return 2
    cfg, err = get_config(args.model, args.effort)
    if err:
        rep.status(f"ERROR: {err} (exit 3)")
        return 3

    out = t.with_name("summary.txt")
    if out.is_file() and out.stat().st_size > 0 and not args.force:
        rep.status("[3/3] verify    -> [PASS] existing summary, skipped")
        rep.status(f"SKIP {out} | {human_bytes(out.stat().st_size)}")
        return 0
    if args.force and out.is_file():
        out.unlink()

    transcript = t.read_text(encoding="utf-8", errors="replace")
    words = len(transcript.split())
    mode = "single-pass" if len(transcript) <= CHUNK_CHARS else "map-reduce"
    rep.status(f"[1/3] info      -> model={cfg['model']}, "
               f"input={words} words, mode={mode}")

    t0 = time.perf_counter()
    text, err, n_chunks = summarize_transcript(cfg, rep, transcript)
    dt = time.perf_counter() - t0
    if err:
        rep.status(f"ERROR: {err} (exit 1)")
        return 1
    if not text.strip():
        rep.status("ERROR: empty summary produced (exit 4)")
        return 4
    rep.status(f"[2/3] summarize -> done ({dt:.1f}s, "
               f"{n_chunks} chunk(s), {len(text.split())} words out)")

    out.write_text(text + "\n", encoding="utf-8")
    rep.status(f"[3/3] verify    -> [PASS] {human_bytes(out.stat().st_size)}, "
               f"{len(text.split())} words")
    rep.status(f"OK  {out} | {dt:.1f}s wall")
    return 0


if __name__ == "__main__":
    sys.exit(main())

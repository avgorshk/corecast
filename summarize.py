#!/usr/bin/env python3
"""CoreCast Stage 3: transcript -> standalone summary via online LLM.

Backends: groq (free tier, OpenAI-compatible) and gigachat (Sber freemium).

Usage: python summarize.py <transcript.txt> [--backend groq|gigachat]
              [--model M] [--force] [-v] [--json-progress]

Keys from environment (or the project .env file):
  groq:     GROQ_API_KEY
  gigachat: GIGACHAT_API_KEY (client secret) + GIGACHAT_CLIENT_ID

Output: <dir>/summary.txt - standalone prose in the transcript's language,
        no references to the source video, no timestamps.

Long transcripts are chunked (map-reduce): each chunk is summarized
separately, the partial summaries are merged into the final text. Chunk
size keeps single requests under free-tier token-per-minute limits.

Exit codes: 0 ok/skip | 1 API failure | 2 input missing |
            3 credentials/config missing | 4 empty output | 5 usage
"""

import argparse
import base64
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from fetch import CLIReporter, JSONReporter, USER_AGENT, human_bytes

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
CERT_FILE = PROJECT_ROOT / "certs" / "russian_trusted_root_ca.cer"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"  # free tier after the Aug-2026 Llama deprecations
GIGACHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_MODEL = "GigaChat"

SYSTEM_PROMPT = (
    "You are a summarization assistant. Produce a standalone summary of the "
    "transcript's major thoughts. Rules:\n"
    "1. Write plain text, no markdown formatting.\n"
    "2. Write in the same language as the transcript.\n"
    "3. The summary must be fully self-contained: never use the words "
    "'author', 'video', 'speaker', 'narrator', 'transcript', 'episode', "
    "'viewer', 'listener', or any reference to timestamps or the source "
    "medium.\n"
    "4. State ideas impersonally: 'the argument is that ...', never 'the "
    "author argues that ...'."
)
MAX_OUT_TOKENS = 1500
CHUNK_CHARS = 16000   # ~5.5-6.5k RU tokens: free-tier TPM limit is 8k per request


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


def gigachat_access_token(client_id, secret):
    """OAuth2 client-credentials against GigaChat (needs the RU CA bundle)."""
    ctx = ssl.create_default_context(cafile=str(CERT_FILE))
    auth = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
    req = urllib.request.Request(
        GIGACHAT_OAUTH_URL,
        data=b"scope=GIGACHAT_API_PERS",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json",
                 "User-Agent": USER_AGENT,
                 "Authorization": f"Basic {auth}",
                 "RqUID": str(uuid.uuid4())})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        data = json.load(r)
    return data.get("access_token")


def setup_backend(backend, model):
    """Return (config_dict, error). config: url, model, headers, ctx."""
    if backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return None, "GROQ_API_KEY is not set (free key: console.groq.com)"
        return {"url": GROQ_URL,
                "model": model or GROQ_MODEL,
                "headers": {"Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "User-Agent": USER_AGENT},
                "ctx": None}, None
    if backend == "gigachat":
        cid = os.environ.get("GIGACHAT_CLIENT_ID")
        secret = os.environ.get("GIGACHAT_API_KEY")
        if not cid or not secret:
            return None, ("GIGACHAT_CLIENT_ID / GIGACHAT_API_KEY not set "
                          "(developers.sber.ru -> GigaChat API)")
        if not CERT_FILE.is_file():
            return None, f"RU CA bundle missing: {CERT_FILE}"
        try:
            token = gigachat_access_token(cid, secret)
        except Exception as e:
            return None, f"GigaChat OAuth failed: {e}"
        if not token:
            return None, "GigaChat OAuth returned no token"
        return {"url": GIGACHAT_URL,
                "model": model or GIGACHAT_MODEL,
                "headers": {"Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                            "User-Agent": USER_AGENT},
                "ctx": ssl.create_default_context(cafile=str(CERT_FILE))}, None
    return None, f"unknown backend: {backend}"


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
    Retries on HTTP 429 using the server's own 'try again in Xs' hint
    (free-tier TPM is a rolling per-minute budget).
    Returns (full_text, error).
    """
    payload = {"model": cfg["model"],
               "messages": messages,
               "max_tokens": MAX_OUT_TOKENS,
               "temperature": 0.2,
               "stream": True}
    data = json.dumps(payload).encode("utf-8")
    est_total = MAX_OUT_TOKENS
    if not quiet:
        rep.phase_start(phase, "[2/3] summarize ")

    for attempt in range(5):
        req = urllib.request.Request(cfg["url"], data=data, method="POST",
                                     headers=cfg["headers"])
        text_parts, chars = [], 0
        try:
            with urllib.request.urlopen(req, timeout=180,
                                        context=cfg.get("ctx")) as resp:
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
                    except json.JSONDecodeError:
                        continue
                    if content:
                        text_parts.append(content)
                        chars += len(content)
                        if not quiet:
                            # rough token estimate for the bar
                            rep.phase_update(phase, min(chars // 3, est_total),
                                             est_total, {"unit": "tok"})
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
                rep.phase_update(phase, est_total, est_total, {"unit": "tok"})
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
             "content": f"Summarize this part of a transcript.\n\n{ch}"},
        ]
        text, err = chat_stream(cfg, messages, rep, quiet=True)
        if err:
            rep.phase_done("summarize")
            return None, f"chunk {i}/{len(chunks)}: {err}", len(chunks)
        parts.append(text)
        time.sleep(2)  # keep sequential requests under the free-tier TPM window
    rep.phase_update("summarize", len(chunks), len(chunks), {"unit": "chunk"})
    rep.phase_done("summarize")

    joined = "\n\n".join(f"--- Part {i + 1} ---\n{p}"
                         for i, p in enumerate(parts))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": ("Combine these partial summaries of the same source into "
                     f"one standalone summary.\n\n{joined}")},
    ]
    text, err = chat_stream(cfg, messages, rep, phase="merge")
    return text, err, len(chunks)


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="summarize.py",
                                 description="CoreCast Stage 3: transcript -> summary")
    ap.add_argument("transcript", help="input transcript file")
    ap.add_argument("--backend", choices=["groq", "gigachat"], default="groq")
    ap.add_argument("--model", help="override the backend's default model")
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
    cfg, err = setup_backend(args.backend, args.model)
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
    rep.status(f"[1/3] info      -> backend={args.backend}, "
               f"model={cfg['model']}, input={words} words, mode={mode}")

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

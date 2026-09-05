#!/usr/bin/env python3
"""CoreCast GUI: pywebview window rendering ui/index.html.

Run:      python gui.py
Or via the CoreCast.lnk shortcut (pythonw.exe, no console window).

The HTML polls Api.get_state(); Api.start() spawns the pipeline worker.
"""

import os
import sys
import atexit
from pathlib import Path

if sys.stdout is None:          # running under pythonw.exe
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import webview                     # noqa: E402  (after the pythonw guard)
from pipeline import PipelineRunner  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
runner = PipelineRunner()


class Api:
    def start(self, url):
        return runner.start(url)

    def get_state(self):
        return runner.state()


def main():
    html = (PROJECT_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    win = webview.create_window("CoreCast", html=html, width=820, height=700,
                                min_size=(700, 560), js_api=Api())
    win.events.closing += runner.cleanup
    atexit.register(runner.cleanup)   # belt and suspenders
    webview.start()


if __name__ == "__main__":
    main()

"""Regression test: platforms that intermittently report formats with
UNKNOWN codecs (observed on VK) must still yield a probeable candidate.

Run: .venv/Scripts/python tests/test_select_format.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fetch import select_format, CLIReporter  # noqa: E402

TMP = Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / "vk3_j.json"


def main():
    if not TMP.is_file():
        print("SKIP: no fresh VK -J dump at", TMP)
        return 0
    meta = json.loads(TMP.read_text(encoding="utf-8"))
    real = meta.get("formats", [])
    if not real:
        print("SKIP: empty format list")
        return 0
    # simulate the failing case: strip codec info from every format
    fake_formats = [
        {"format_id": f.get("format_id"), "url": f.get("url"),
         "vcodec": None, "acodec": None, "height": f.get("height"),
         "tbr": f.get("tbr"), "ext": f.get("ext")}
        for f in real if f.get("url")
    ]
    sel, reason = select_format({"formats": fake_formats}, CLIReporter(False), verbose=True)
    print("selector:", sel)
    print("reason:", reason)
    if sel is None:
        print("FAIL: unknown-codec formats were rejected")
        return 1
    print("PASS: unknown-codec formats accepted as candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())

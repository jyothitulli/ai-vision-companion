"""Copy pipeline outputs that look like failures into the dataset inbox.

This never stores images unless --copy-images is passed AND the operator
confirms this is an explicit dataset collection session.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "datasets" / "accessibility" / "annotations" / "inbox"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-json", type=Path, required=True, help="DetectionEvent-like JSON")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    INBOX.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.event_json.read_text(encoding="utf-8"))
    if args.copy_images:
        raise SystemExit("Image copy is disabled by default. Use a dedicated collection machine with informed consent.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = INBOX / f"failure_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote metadata-only failure case to {out}")


if __name__ == "__main__":
    main()

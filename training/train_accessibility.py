"""Fine-tuning entry point. Does not invent results.

Example after images and YOLO labels exist:

  yolo detect train data=datasets/accessibility/data.yaml model=yolo11n.pt epochs=50 imgsz=640
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "accessibility" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run or not args.data.exists():
        print("Fine-tuning is not started.")
        print(f"Expected dataset yaml: {args.data}")
        print("Collect failure cases into datasets/accessibility/{train,val,test}/images and labels, then rerun without --dry-run.")
        return
    from ultralytics import YOLO

    YOLO(args.model).train(data=str(args.data), epochs=args.epochs, imgsz=640, project=str(ROOT / "training" / "runs"))


if __name__ == "__main__":
    main()

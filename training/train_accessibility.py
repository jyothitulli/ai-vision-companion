"""Accessibility Model Training and Evaluation Pipeline.

Supports dataset validation, YOLO-format training, validation, testing,
and checkpoint management for dedicated accessibility classes.

Usage:
    python training/train_accessibility.py --validate-only
    python training/train_accessibility.py --data datasets/accessibility/data.yaml --model yolo11n.pt --epochs 30
    python training/train_accessibility.py --mode val --weights training/runs/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "accessibility" / "data.yaml"


def validate_dataset(data_yaml_path: Path) -> tuple[bool, str, dict[str, int]]:
    """Validate YOLO-format dataset structure, label bounds, and class distributions."""
    if not data_yaml_path.exists():
        return False, f"data.yaml not found at {data_yaml_path}", {}

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            return False, f"Failed to parse data.yaml: {e}", {}

    base_dir = data_yaml_path.parent
    counts = {"train": 0, "val": 0, "test": 0}
    splits = {
        "train": config.get("train", "images/train"),
        "val": config.get("val", "images/val"),
        "test": config.get("test", "images/test"),
    }

    names = config.get("names", {})
    if not names:
        return False, "No class names defined in data.yaml", {}

    max_cls_id = max(names.keys()) if isinstance(names, dict) else len(names) - 1

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split_name, rel_path in splits.items():
        img_dir = (base_dir / rel_path).resolve()
        if not img_dir.exists():
            # Check alternative layout (e.g. train/images)
            alt_dir = base_dir / split_name / "images"
            if alt_dir.exists():
                img_dir = alt_dir
            else:
                continue

        images = [p for p in img_dir.glob("*") if p.suffix.lower() in valid_extensions]
        counts[split_name] = len(images)

        # Check label pairing
        label_dir = img_dir.parent / "labels"
        if not label_dir.exists():
            alt_lbl = base_dir / "labels" / split_name
            if alt_lbl.exists():
                label_dir = alt_lbl

        for img in images:
            lbl_file = label_dir / f"{img.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    for line_idx, line in enumerate(lf):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if cls_id > max_cls_id:
                            return False, f"Invalid class ID {cls_id} in {lbl_file.name}:{line_idx}", counts
                        coords = [float(x) for x in parts[1:5]]
                        for c in coords:
                            if c < 0.0 or c > 1.0:
                                return False, f"Normalized bbox coordinate out of bounds [0, 1] in {lbl_file.name}", counts

    total_images = sum(counts.values())
    if total_images == 0:
        return True, "Dataset structure valid, but 0 images currently populated (inbox ready).", counts

    return True, f"Dataset valid with {counts['train']} train, {counts['val']} val, {counts['test']} test images.", counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Accessibility Model Training & Evaluation Pipeline")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model checkpoint or architecture")
    parser.add_argument("--weights", default=None, help="Trained weights for evaluation/testing")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--mode", default="train", choices=["train", "val", "test"], help="Execution mode")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("ACCESSIBILITY MODEL TRAINING & DATASET PIPELINE")
    print("=" * 60)
    print(f"Data config : {args.data}")
    print(f"Mode        : {args.mode}")

    is_valid, msg, counts = validate_dataset(args.data)
    print(f"Validation  : {'PASSED' if is_valid else 'FAILED'} - {msg}")
    print(f"Counts      : Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}")
    print("=" * 60)

    if args.validate_only or not is_valid:
        sys.exit(0 if is_valid else 1)

    if counts.get("train", 0) == 0 and args.mode == "train":
        print("\n[INFO] No training images detected in datasets/accessibility/.")
        print("To train a fine-tuned model for specialized accessibility classes:")
        print("  1. Add photographic images to datasets/accessibility/images/train/ (and val/)")
        print("  2. Add corresponding YOLO-format labels (.txt) to datasets/accessibility/labels/train/")
        print("  3. Rerun: python training/train_accessibility.py --epochs 30")
        print("\nSkipping training execution to prevent hallucinating trained checkpoints.")
        sys.exit(0)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run in active venv.")
        sys.exit(1)

    runs_dir = ROOT / "training" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        print(f"\nInitiating YOLO fine-tuning on {args.model}...")
        model = YOLO(args.model)
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=str(runs_dir),
            name="accessibility_yolo11",
            exist_ok=True,
        )
        print("\nTraining completed successfully.")
        print(f"Checkpoints saved in: {runs_dir / 'accessibility_yolo11' / 'weights'}")
    elif args.mode in {"val", "test"}:
        weights_path = args.weights or str(runs_dir / "accessibility_yolo11" / "weights" / "best.pt")
        if not Path(weights_path).exists():
            print(f"[ERROR] Trained weights not found at: {weights_path}")
            print("Train the model first or specify --weights <path>.")
            sys.exit(1)
        model = YOLO(weights_path)
        metrics = model.val(data=str(args.data), split=args.mode, imgsz=args.imgsz)
        print(f"\n{args.mode.upper()} Results:")
        print(f"  mAP@50    : {metrics.box.map50:.4f}")
        print(f"  mAP@50:95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

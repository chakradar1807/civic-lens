"""Train YOLOv8 on the merged CivicLens dataset, evaluate it and copy best.pt to models/.

Usage (Google Colab / Kaggle with a GPU):
    pip install ultralytics
    python training/train_yolo.py --data training/data.yaml --epochs 50 --imgsz 640
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="training/data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt is small enough for CPU deployment")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/civiclens")
    p.add_argument("--name", default="yolov8n")
    p.add_argument("--out", default="models/best.pt")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.name, exist_ok=True, patience=15)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training finished but {best} was not found")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)

    # Evaluation numbers for your report (ultralytics also saves confusion_matrix.png,
    # PR curves and val predictions in the run folder).
    metrics = YOLO(args.out).val(data=args.data, imgsz=args.imgsz)
    summary = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class_mAP50-95": {metrics.names[i]: round(float(v), 4) for i, v in enumerate(metrics.box.maps)},
    }
    out = Path(args.project) / args.name / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWeights copied to {args.out}. Evaluation plots are in {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()

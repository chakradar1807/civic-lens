"""Merge several YOLO-format datasets (e.g. Roboflow exports) into one 4-class dataset.

Each source dataset has its own class ids. Describe the mapping on the command line:

    python training/merge_datasets.py \
        --source datasets/potholes   --map pothole=0 \
        --source datasets/garbage    --map garbage=0 \
        --source datasets/streetlamp --map damaged_streetlight=1 \
        --source datasets/floods     --map waterlogging=0 \
        --out datasets/civiclens --val-split 0.15

`--map NAME=SRC_ID[,NAME=SRC_ID...]` says: source class id SRC_ID becomes our class NAME.
Source class ids that are not mapped are dropped (their boxes are removed from the label files).
Expected source layout: <source>/{train,valid,test}/{images,labels}  OR  <source>/{images,labels}.
"""
import argparse
import random
import shutil
from pathlib import Path

CLASSES = ["pothole", "garbage", "damaged_streetlight", "waterlogging"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(src: Path):
    pairs = []
    for img in src.rglob("*"):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        label = Path(str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if label.is_file():
            pairs.append((img, label))
    return pairs


def remap(label_path: Path, mapping: dict):
    lines = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(float(parts[0]))
        except ValueError:
            continue
        if src_id in mapping:
            lines.append(" ".join([str(mapping[src_id])] + parts[1:]))
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--map", action="append", required=True, help="NAME=SRC_ID[,NAME=SRC_ID]")
    p.add_argument("--out", default="datasets/civiclens")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-empty", action="store_true", help="keep images with no mapped boxes as background negatives")
    a = p.parse_args()
    if len(a.source) != len(a.map):
        raise SystemExit("Give one --map for every --source")

    random.seed(a.seed)
    items = []
    for idx, (src, mp) in enumerate(zip(a.source, a.map)):
        mapping = {}
        for chunk in mp.split(","):
            name, sid = chunk.split("=")
            if name.strip() not in CLASSES:
                raise SystemExit(f"Unknown class '{name}'. Use one of {CLASSES}")
            mapping[int(sid)] = CLASSES.index(name.strip())
        pairs = find_pairs(Path(src))
        print(f"{src}: {len(pairs)} labelled images found")
        for img, lab in pairs:
            lines = remap(lab, mapping)
            if lines or a.keep_empty:
                items.append((idx, img, lines))

    if not items:
        raise SystemExit("No images were collected - check the --source paths and --map ids")
    random.shuffle(items)
    n_val = max(1, int(len(items) * a.val_split))
    out = Path(a.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    for i, (src_idx, img, lines) in enumerate(items):
        split = "val" if i < n_val else "train"
        stem = f"s{src_idx}_{i:06d}"
        shutil.copy(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        for ln in lines:
            counts[CLASSES[int(ln.split()[0])]] += 1
    print(f"\nMerged {len(items)} images ({n_val} val) into {out}")
    print("Boxes per class:", counts)
    low = [c for c, n in counts.items() if n < 200]
    if low:
        print("WARNING: fewer than 200 boxes for", low, "- collect more data for these classes.")


if __name__ == "__main__":
    main()

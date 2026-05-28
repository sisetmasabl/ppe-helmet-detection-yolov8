"""
Análisis exploratorio del dataset YOLO:
  - Distribución de instancias por clase y por split (desbalance de clases).
  - Distribución de tamaños de bounding box (proxy de distancia: bbox pequeño = lejano).

Genera:
  - outputs/dataset_class_distribution.png
  - outputs/dataset_bbox_size_distribution.png
  - outputs/dataset_summary.json

Uso:
    python src/analyze_dataset.py --data data/<dataset>/data.yaml
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Umbrales de tamaño relativo (raíz del área normalizada = tamaño lineal relativo)
SMALL_T = 0.08   # < 8% del lado de la imagen  -> lejano / objeto pequeño
LARGE_T = 0.25   # > 25% del lado de la imagen -> cercano / objeto grande


def labels_dir_for(images_path: Path) -> Path:
    return images_path.parent / "labels"


def resolve_split_dirs(data_yaml: Path):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = data_yaml.parent
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    splits = {}
    for split in ("train", "val", "test"):
        key = "val" if split == "val" else split
        val = cfg.get(key)
        if not val:
            continue
        p = (base / val).resolve()
        # data.yaml suele apuntar a .../images
        if p.name != "images" and (p / "images").exists():
            p = p / "images"
        splits[split] = p
    return cfg, names, splits


def size_bucket(w, h):
    rel = math.sqrt(max(w, 0) * max(h, 0))
    if rel < SMALL_T:
        return "small"   # lejano
    if rel > LARGE_T:
        return "large"   # cercano
    return "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg, names, splits = resolve_split_dirs(Path(args.data).resolve())

    per_split_class = defaultdict(lambda: defaultdict(int))   # split -> class_id -> count
    size_by_class = defaultdict(lambda: defaultdict(int))      # class -> bucket -> count
    size_overall = defaultdict(int)
    img_counts = {}

    for split, img_dir in splits.items():
        lbl_dir = labels_dir_for(img_dir)
        n_imgs = 0
        if img_dir.exists():
            n_imgs = sum(1 for _ in img_dir.glob("*.*"))
        img_counts[split] = n_imgs
        if not lbl_dir.exists():
            continue
        for lf in lbl_dir.glob("*.txt"):
            for line in lf.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                w, h = float(parts[3]), float(parts[4])
                per_split_class[split][cid] += 1
                b = size_bucket(w, h)
                cname = names[cid] if names and cid < len(names) else str(cid)
                size_by_class[cname][b] += 1
                size_overall[b] += 1

    # ---- Resumen JSON ----
    summary = {
        "classes": names,
        "images_per_split": img_counts,
        "instances_per_class_per_split": {
            s: {(names[c] if names and c < len(names) else str(c)): n for c, n in d.items()}
            for s, d in per_split_class.items()
        },
        "size_distribution_overall": dict(size_overall),
        "size_distribution_by_class": {k: dict(v) for k, v in size_by_class.items()},
        "size_thresholds": {"small_max_rel": SMALL_T, "large_min_rel": LARGE_T},
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- Plot 1: distribución de clases por split ----
    all_classes = sorted({c for d in per_split_class.values() for c in d})
    labels = [names[c] if names and c < len(names) else str(c) for c in all_classes]
    fig, ax = plt.subplots(figsize=(max(6, len(all_classes) * 1.2), 4))
    x = range(len(all_classes))
    width = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        if split not in per_split_class:
            continue
        vals = [per_split_class[split].get(c, 0) for c in all_classes]
        ax.bar([xi + i * width for xi in x], vals, width, label=split)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Instancias")
    ax.set_title("Distribución de clases por split (desbalance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "dataset_class_distribution.png", dpi=130)
    plt.close(fig)

    # ---- Plot 2: distribución de tamaños (proxy de distancia) ----
    buckets = ["small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(buckets, [size_overall.get(b, 0) for b in buckets],
           color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Instancias")
    ax.set_title("Tamaño de objeto (small=lejano, large=cercano)")
    for i, b in enumerate(buckets):
        ax.text(i, size_overall.get(b, 0), str(size_overall.get(b, 0)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "dataset_bbox_size_distribution.png", dpi=130)
    plt.close(fig)

    print(f"\nGuardado en: {OUT}")


if __name__ == "__main__":
    main()

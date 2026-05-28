"""
Evaluación cuantitativa y análisis de robustez del modelo YOLOv8.

Produce los entregables 2 y 3:
  (2) mAP@0.5 y mAP@0.5:0.95 sobre el conjunto de prueba, con desglose por clase.
  (3) Análisis de robustez:
        - por TAMAÑO de objeto (proxy de distancia): small=lejano / medium / large=cercano
        - por nivel de OCLUSIÓN (proxy semiautomático): bajo / medio / alto
      Se reporta el RECALL@IoU0.5 (en el punto de operación conf=0.25) por grupo,
      que responde directamente a "¿detecta el casco a larga distancia?".

Salidas en outputs/:
  - metrics_overall.json          (mAP global y por clase)
  - robustness_by_distance.csv/.png
  - robustness_by_occlusion.csv/.png
  - robustness_summary.json

Uso:
    python src/evaluate.py --data data/<dataset>/data.yaml --weights runs/detect/ppe_yolov8s/weights/best.pt
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Proxy de distancia (tamaño lineal relativo = sqrt(area normalizada))
SMALL_T = 0.08
LARGE_T = 0.25
# Proxy de oclusión (máx. IoU de un bbox con cualquier otro del mismo fotograma)
OCC_LOW = 0.10
OCC_HIGH = 0.35
CONF_OP = 0.25   # punto de operación para el análisis de recall
IOU_MATCH = 0.5


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
        if p.name != "images" and (p / "images").exists():
            p = p / "images"
        splits[split] = p
    return cfg, names, splits


def size_bucket(w_rel, h_rel):
    rel = math.sqrt(max(w_rel, 0) * max(h_rel, 0))
    if rel < SMALL_T:
        return "small"
    if rel > LARGE_T:
        return "large"
    return "medium"


def occ_bucket(max_iou):
    if max_iou < OCC_LOW:
        return "low"
    if max_iou > OCC_HIGH:
        return "high"
    return "medium"


def iou_xyxy(a, b):
    # a: [N,4], b: [M,4] -> [N,M]
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_gt(label_path: Path, w: int, h: int):
    """Devuelve listas paralelas: cls[], box_xyxy[], size_bucket[], occ_bucket[]."""
    cls, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
            x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
            cls.append(c); boxes.append([x1, y1, x2, y2])
    # tamaño desde dimensiones normalizadas
    sbk = []
    for box in boxes:
        wr = (box[2] - box[0]) / w; hr = (box[3] - box[1]) / h
        sbk.append(size_bucket(wr, hr))
    # oclusión: máx IoU con otro GT del mismo fotograma
    obk = []
    if boxes:
        m = iou_xyxy(boxes, boxes)
        np.fill_diagonal(m, 0.0)
        maxiou = m.max(axis=1) if m.shape[1] > 1 else np.zeros(len(boxes))
        obk = [occ_bucket(v) for v in maxiou]
    return cls, boxes, sbk, obk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, names, splits = resolve_split_dirs(Path(args.data).resolve())
    if args.split not in splits:
        args.split = "val"
    img_dir = splits[args.split]
    lbl_dir = img_dir.parent / "labels"

    model = YOLO(args.weights)

    # ---------- (2) Métricas estándar mAP por clase ----------
    print("== Validación estándar (mAP) ==")
    metrics = model.val(data=args.data, split=args.split, device=args.device,
                        project=str(OUT), name="val_metrics", exist_ok=True, plots=True)
    per_class = {}
    try:
        for i, c in enumerate(metrics.box.ap_class_index):
            cname = names[c] if names and c < len(names) else str(c)
            per_class[cname] = {
                "AP50": float(metrics.box.ap50[i]),
                "AP50_95": float(metrics.box.ap[i]),
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
            }
    except Exception as e:  # noqa
        print("aviso por-clase:", e)
    overall = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
        "split": args.split,
    }
    (OUT / "metrics_overall.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    print(json.dumps(overall, indent=2, ensure_ascii=False))

    # ---------- (3) Análisis estratificado (distancia y oclusión) ----------
    # matched/total por bucket (global) y por clase
    dist_stats = {b: {"tp": 0, "total": 0, "conf_sum": 0.0} for b in ["small", "medium", "large"]}
    occ_stats = {b: {"tp": 0, "total": 0, "conf_sum": 0.0} for b in ["low", "medium", "high"]}
    dist_by_class = defaultdict(lambda: {b: {"tp": 0, "total": 0} for b in ["small", "medium", "large"]})

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [p for p in img_dir.glob("*.*") if p.suffix.lower() in exts]
    print(f"\n== Análisis estratificado sobre {len(images)} imágenes ({args.split}) ==")

    for k, ip in enumerate(images):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_cls, gt_box, gt_sz, gt_oc = load_gt(lbl_dir / (ip.stem + ".txt"), w, h)
        res = model.predict(img, conf=CONF_OP, iou=0.6, device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            pcf = res.boxes.conf.cpu().numpy()
        else:
            pb = np.zeros((0, 4)); pc = np.array([], int); pcf = np.array([])

        used = np.zeros(len(pb), dtype=bool)
        ious = iou_xyxy(gt_box, pb) if len(gt_box) and len(pb) else np.zeros((len(gt_box), len(pb)))
        for gi in range(len(gt_box)):
            sb, ob, gc = gt_sz[gi], gt_oc[gi], gt_cls[gi]
            dist_stats[sb]["total"] += 1
            occ_stats[ob]["total"] += 1
            cname = names[gc] if names and gc < len(names) else str(gc)
            dist_by_class[cname][sb]["total"] += 1
            best_j, best_iou = -1, IOU_MATCH
            for j in range(len(pb)):
                if used[j] or pc[j] != gc:
                    continue
                if ious[gi, j] >= best_iou:
                    best_iou = ious[gi, j]; best_j = j
            if best_j >= 0:
                used[best_j] = True
                dist_stats[sb]["tp"] += 1
                dist_stats[sb]["conf_sum"] += float(pcf[best_j])
                occ_stats[ob]["tp"] += 1
                occ_stats[ob]["conf_sum"] += float(pcf[best_j])
                dist_by_class[cname][sb]["tp"] += 1
        if (k + 1) % 100 == 0:
            print(f"  procesadas {k+1}/{len(images)}")

    def finalize(stats):
        out = {}
        for b, s in stats.items():
            rec = s["tp"] / s["total"] if s["total"] else 0.0
            mc = s["conf_sum"] / s["tp"] if s["tp"] else 0.0
            out[b] = {"recall": rec, "tp": s["tp"], "total": s["total"], "mean_conf": mc}
        return out

    dist_final = finalize(dist_stats)
    occ_final = finalize(occ_stats)
    robustness = {
        "operating_point_conf": CONF_OP, "iou_match": IOU_MATCH,
        "distance": dist_final, "occlusion": occ_final,
        "distance_thresholds": {"small_max_rel": SMALL_T, "large_min_rel": LARGE_T},
        "occlusion_thresholds": {"low_max_iou": OCC_LOW, "high_min_iou": OCC_HIGH},
        "distance_by_class": {c: {b: (v["tp"] / v["total"] if v["total"] else 0.0)
                                   for b, v in d.items()} for c, d in dist_by_class.items()},
    }
    (OUT / "robustness_summary.json").write_text(json.dumps(robustness, indent=2, ensure_ascii=False))
    print(json.dumps(robustness, indent=2, ensure_ascii=False))

    # CSVs
    for fname, final, order in [
        ("robustness_by_distance.csv", dist_final, ["small", "medium", "large"]),
        ("robustness_by_occlusion.csv", occ_final, ["low", "medium", "high"]),
    ]:
        with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["bucket", "recall", "tp", "total", "mean_conf"])
            for b in order:
                r = final[b]
                wcsv.writerow([b, f"{r['recall']:.4f}", r["tp"], r["total"], f"{r['mean_conf']:.4f}"])

    # Plots
    _bar(dist_final, ["small", "medium", "large"],
         ["small\n(lejano)", "medium", "large\n(cercano)"],
         "Recall por distancia (tamaño de objeto)", OUT / "robustness_by_distance.png",
         ["#d62728", "#ff7f0e", "#2ca02c"])
    _bar(occ_final, ["low", "medium", "high"],
         ["bajo", "medio", "alto"],
         "Recall por nivel de oclusión", OUT / "robustness_by_occlusion.png",
         ["#2ca02c", "#ff7f0e", "#d62728"])
    print(f"\nGuardado en: {OUT}")


def _bar(final, order, labels, title, path, colors):
    vals = [final[b]["recall"] for b in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Recall @ IoU0.5 (conf=0.25)")
    ax.set_title(title)
    for bar, b in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{final[b]['recall']:.2f}\n(n={final[b]['total']})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()

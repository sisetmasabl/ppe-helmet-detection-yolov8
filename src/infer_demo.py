"""
Demostración del sistema de verificación de cumplimiento sobre imágenes/video
NO vistos durante el entrenamiento (entregable 4).

- Ejecuta el modelo YOLOv8 sobre una carpeta de imágenes o un archivo de video.
- Aplica el verificador de cumplimiento basado en reglas (compliance_verifier.py).
- Guarda fotogramas anotados, un clip de video corto y un CSV con la tasa de
  cumplimiento por fotograma.

Uso:
    # Sobre el split de test (imágenes no vistas):
    python src/infer_demo.py --weights runs/detect/ppe_yolov8s/weights/best.pt \
        --source data/<dataset>/test/images --max 60 --make-video

    # Sobre un video:
    python src/infer_demo.py --weights best.pt --source clip.mp4 --make-video
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from compliance_verifier import ComplianceVerifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_W, VIDEO_H = 1280, 720   # lienzo fijo del clip (imágenes de tamaño heterogéneo)


def letterbox_to(img, dst_w=VIDEO_W, dst_h=VIDEO_H, color=(0, 0, 0)):
    """Reescala manteniendo aspecto y rellena hasta (dst_w, dst_h).

    Necesario porque el conjunto puede tener imágenes de tamaños muy distintos
    (p. ej. SHWD): el VideoWriter exige un tamaño de fotograma constante.
    """
    h, w = img.shape[:2]
    s = min(dst_w / w, dst_h / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((dst_h, dst_w, 3), color, dtype=np.uint8)
    x0, y0 = (dst_w - nw) // 2, (dst_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def iter_frames(source: Path, max_n: int):
    """Genera (nombre, frame_bgr) desde carpeta de imágenes o video."""
    if source.is_dir():
        files = sorted(p for p in source.glob("*.*") if p.suffix.lower() in IMG_EXTS)
        if max_n:
            files = files[:max_n]
        for p in files:
            img = cv2.imread(str(p))
            if img is not None:
                yield p.stem, img
    else:
        cap = cv2.VideoCapture(str(source))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_n and i >= max_n):
                break
            yield f"frame_{i:05d}", frame
            i += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="carpeta de imágenes o archivo de video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max", type=int, default=60, help="máx. fotogramas a procesar")
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--outdir", default=str(OUT / "demo"))
    args = ap.parse_args()

    from ultralytics import YOLO
    outdir = Path(args.outdir)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, list) else [model.names[i] for i in sorted(model.names)]
    verifier = ComplianceVerifier(names, conf_threshold=args.conf)
    print("Clases 'person':", [names[i] for i in verifier.person_ids])
    print("Clases 'helmet':", [names[i] for i in verifier.helmet_ids])

    rows = []
    writer = None
    n_violation_frames = 0
    for name, frame in iter_frames(Path(args.source), args.max):
        res = model.predict(frame, conf=args.conf, iou=0.6, device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
        else:
            boxes, clss, confs = np.zeros((0, 4)), np.array([], int), np.array([])

        fr = verifier.process_frame(frame, boxes, clss, confs)
        out_path = frames_dir / f"{name}.jpg"
        cv2.imwrite(str(out_path), fr.annotated)
        rate = fr.compliance_rate
        if fr.counts["violation"] > 0:
            n_violation_frames += 1
        rows.append({
            "frame": name,
            "compliance_rate": "" if rate is None else round(rate, 4),
            "persons": fr.counts["persons"],
            "compliant": fr.counts["compliant"],
            "violation": fr.counts["violation"],
            "helmets": fr.counts["helmets"],
        })

        if args.make_video:
            if writer is None:
                writer = cv2.VideoWriter(str(outdir / "demo_clip.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (VIDEO_W, VIDEO_H))
            writer.write(letterbox_to(fr.annotated))

    if writer is not None:
        writer.release()

    with open(outdir / "compliance_per_frame.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["frame", "compliance_rate", "persons", "compliant", "violation", "helmets"])
        wcsv.writeheader(); wcsv.writerows(rows)

    rates = [r["compliance_rate"] for r in rows if r["compliance_rate"] != ""]
    summary = {
        "frames_processed": len(rows),
        "frames_with_violations": n_violation_frames,
        "mean_compliance_rate": round(float(np.mean(rates)), 4) if rates else None,
        "person_classes": [names[i] for i in verifier.person_ids],
        "helmet_classes": [names[i] for i in verifier.helmet_ids],
    }
    (outdir / "demo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nFotogramas anotados: {frames_dir}")
    if args.make_video:
        print(f"Video demo: {outdir / 'demo_clip.mp4'}")


if __name__ == "__main__":
    main()

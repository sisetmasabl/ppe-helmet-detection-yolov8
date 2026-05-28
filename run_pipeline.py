"""
Pipeline de extremo a extremo (COMANDO ÚNICO) para el proyecto de detección de
cumplimiento de EPP/casco con YOLOv8, sobre el dataset recomendado **SHWD**
(Safety-Helmet-Wearing-Dataset), que incluye ejemplos explícitos de "sin casco".

Ejecuta en orden:
  1. Descarga SHWD desde Google Drive (gdown, SIN API key) y lo extrae.
  2. Convierte VOC -> YOLO y construye splits train/val/test (data/shwd_yolo).
  3. Análisis exploratorio del dataset (clases, tamaños/distancia).
  4. Entrenamiento YOLOv8 (imgsz=640, 100 épocas).
  5. Evaluación cuantitativa + análisis de robustez (oclusión, distancia).
  6. Demostración del verificador de cumplimiento sobre imágenes no vistas (test).

No requiere credenciales (a diferencia del flujo Roboflow original).

Uso:
    python run_pipeline.py                 # pipeline completo
    python run_pipeline.py --skip-train    # reutiliza pesos ya entrenados
    python run_pipeline.py --epochs 100 --model yolov8s.pt --imgsz 640
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
PY = sys.executable


def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="8 por defecto (GPU 8GB)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ppe_yolov8s_shwd")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    voc_dir = DATA / "SHWD" / "VOC2028"
    data_yaml = DATA / "shwd_yolo" / "data.yaml"

    # 1. Descargar SHWD (sin API key)
    if not args.skip_download and not (voc_dir / "JPEGImages").exists():
        run([PY, SRC / "download_shwd.py"])

    # 2. Convertir VOC -> YOLO + splits
    if not data_yaml.exists():
        run([PY, SRC / "prepare_shwd.py", "--src", voc_dir, "--dst", DATA / "shwd_yolo"])
    if not data_yaml.exists():
        sys.exit(f"No se encontró {data_yaml}. ¿Falló la conversión de SHWD?")

    # 3. Analizar dataset
    run([PY, SRC / "analyze_dataset.py", "--data", data_yaml])

    # 4. Entrenar
    weights = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    if not args.skip_train:
        run([PY, SRC / "train.py", "--data", data_yaml, "--model", args.model,
             "--epochs", args.epochs, "--imgsz", args.imgsz, "--batch", args.batch,
             "--device", args.device, "--name", args.name])
    if not weights.exists():
        sys.exit(f"No se encontraron pesos entrenados en {weights}")

    # Copiar mejor modelo a weights/
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(weights, wdir / "best.pt")
    print(f"Modelo copiado a {wdir / 'best.pt'}")

    # 5. Evaluar + robustez
    run([PY, SRC / "evaluate.py", "--data", data_yaml, "--weights", weights,
         "--split", "test", "--device", args.device])

    # 6. Demo del verificador sobre test (no visto en entrenamiento)
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    test_imgs = cfg.get("test") or cfg.get("val")
    run([PY, SRC / "infer_demo.py", "--weights", weights, "--source", test_imgs,
         "--max", 80, "--make-video", "--device", args.device])

    print("\n=== PIPELINE COMPLETO ===")
    print("Resultados en outputs/  |  Modelo en weights/best.pt")


if __name__ == "__main__":
    main()

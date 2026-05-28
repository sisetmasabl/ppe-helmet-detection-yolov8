"""
Entrenamiento de YOLOv8 para detección de cumplimiento de EPP (casco) en obras.

Decisiones de entrenamiento (documentadas en el README y el informe):
  - Backbone preentrenado en COCO (yolov8s.pt) -> transfer learning.
  - Resolución de entrada: 640 px (tamaño "mediano"), buen balance para una
    GPU de 8 GB y suficiente para objetos pequeños a media/larga distancia.
  - 100 épocas.
  - Mosaic augmentation ACTIVADO, desactivado en las últimas 10 épocas
    (close_mosaic=10) para estabilizar la localización fina al final.
  - YOLOv8 es anchor-free (no usa anclas predefinidas), lo que favorece la
    detección de objetos pequeños/lejanos.
  - Semilla fija (42) para reproducibilidad.

Uso:
    python src/train.py --data data/<dataset>/data.yaml --epochs 100 --imgsz 640
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Entrenar YOLOv8 para detección de EPP/casco")
    p.add_argument("--data", type=str, required=True, help="Ruta a data.yaml")
    p.add_argument("--model", type=str, default="yolov8s.pt", help="Pesos base (backbone preentrenado)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640, help="Resolución de entrada (tamaño mediano)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="0", help="'0' GPU, 'cpu' o '0,1'")
    p.add_argument("--name", type=str, default="ppe_yolov8s")
    p.add_argument("--project", type=str, default=str(ROOT / "runs" / "detect"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=120, help="Alto = sin early stopping (100 épocas completas)")
    p.add_argument("--close-mosaic", type=int, default=10, help="Épocas finales sin mosaic")
    p.add_argument("--workers", type=int, default=2,
                   help="Workers del DataLoader. Bajo (2) para limitar uso de RAM del sistema.")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("AVISO: CUDA no disponible, se entrenará en CPU (lento).")
        device = "cpu"
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=args.project,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        close_mosaic=args.close_mosaic,
        # --- Estrategia de aumento de datos (documentada) ---
        mosaic=1.0,        # mosaic ON (ayuda a objetos pequeños y contexto)
        mixup=0.0,         # mixup desactivado (puede confundir el clasificador de cumplimiento)
        copy_paste=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # variación de color (robustez a iluminación de obra)
        degrees=0.0, translate=0.1, scale=0.5,  # 'scale' simula distintas distancias (cerca/lejos)
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        plots=True,        # genera curvas y matrices de confusión
        val=True,
        verbose=True,
    )

    print("\nEntrenamiento finalizado.")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Mejor modelo: {best}")


if __name__ == "__main__":
    main()

# Detección de Cumplimiento de EPP (Casco) en Obras de Construcción — YOLOv8

Sistema de visión por computador que entrena y evalúa un modelo **YOLOv8** para
detectar trabajadores y verificar si **usan o no casco de seguridad**, con énfasis
en la detección a **corta y larga distancia** (objetos pequeños), robustez ante
**oclusión** y un **verificador de cumplimiento basado en reglas**.

> Pregunta 3 — Detección de Cumplimiento de Normas de Seguridad en Obras de
> Construcción (YOLO). Reto: oclusión severa, objetos pequeños (trabajadores
> lejanos) y desbalance de clases (incumplimientos escasos).

## Resultados (conjunto de prueba)

| Métrica | Valor |
|---|---|
| mAP@0.5 | **0.942** |
| mAP@0.5:0.95 | **0.614** |
| Recall objetos *small* (lejanos) | **0.927** |
| Recall objetos *large* (cercanos) | **0.978** |

Dataset: **SHWD** (Safety-Helmet-Wearing-Dataset), clases `helmet` (con casco) / `head` (sin casco).

*(Tablas completas por clase y por grupo de oclusión/distancia en `outputs/`. Informe completo en formato NeurIPS: `report/neurips/main.pdf`.)*

## Estructura del repositorio

```
YOLO8/
├── run_pipeline.py          # COMANDO ÚNICO (sin API key): descarga SHWD -> convierte -> analiza -> entrena -> evalúa -> demo
├── requirements.txt
├── src/
│   ├── download_shwd.py     # descarga SHWD desde Google Drive (gdown, SIN credenciales)
│   ├── prepare_shwd.py      # convierte SHWD (VOC) -> YOLO + splits + data.yaml
│   ├── download_data.py     # (alternativa) descarga desde Roboflow vía ROBOFLOW_API_KEY
│   ├── analyze_dataset.py   # distribución de clases y tamaños (desbalance / distancia)
│   ├── train.py             # entrenamiento YOLOv8 (imgsz=640, 100 épocas, mosaic)
│   ├── evaluate.py          # mAP por clase + robustez por distancia y oclusión
│   ├── compliance_verifier.py  # verificador de cumplimiento basado en reglas
│   └── infer_demo.py        # demo: fotogramas anotados + clip de video
├── notebooks/
│   └── YOLOv8_EPP_Colab.ipynb   # notebook estilo Google Colab
├── outputs/                 # métricas, gráficos, demo (generado)
├── weights/                 # best.pt final (generado)
└── report/
    ├── neurips/             # INFORME EN FORMATO NEURIPS (LaTeX): main.tex, main.pdf, references.bib, neurips_2023.sty
    ├── REPORT.md            # versión narrativa (Markdown) -> report/REPORT.pdf
    ├── build_pdf.py         # genera REPORT.pdf desde REPORT.md (fpdf2)
    └── related_work.md      # referencias verificadas
```

## Requisitos del entorno

- Python 3.10–3.13
- GPU NVIDIA recomendada. **Para GPUs Blackwell (RTX 50xx) se requiere PyTorch con CUDA 12.8.**

### Instalación (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch con CUDA 12.8 (necesario para RTX 50xx; para otras GPU/CPU ver nota)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Resto de dependencias
pip install -r requirements.txt
```

> **Otras GPUs / CPU:** sustituye el índice `cu128` por el que corresponda a tu
> CUDA (p. ej. `cu121`) o instala la versión CPU desde pypi. Ver
> https://pytorch.org/get-started/locally/

### Linux / Google Colab

```bash
pip install -r requirements.txt    # en Colab torch ya viene con CUDA
```

## Reproducción con un solo comando

**No requiere credenciales:** el dataset **SHWD** se descarga automáticamente desde
Google Drive (vía `gdown`).

```bash
python run_pipeline.py
```

Esto descarga SHWD (~1.1 GB), lo convierte de VOC a YOLO, analiza clases/tamaños,
entrena YOLOv8 (100 épocas, imgsz 640), evalúa (mAP + robustez por distancia y
oclusión) y genera la demostración del verificador de cumplimiento. Para reutilizar
pesos ya entrenados: `python run_pipeline.py --skip-train`.

### Uso por etapas (opcional)

```bash
python src/download_shwd.py
python src/prepare_shwd.py  --src data/SHWD/VOC2028 --dst data/shwd_yolo
python src/analyze_dataset.py --data data/shwd_yolo/data.yaml
python src/train.py    --data data/shwd_yolo/data.yaml --epochs 100 --imgsz 640 --batch 8 --workers 2
python src/evaluate.py --data data/shwd_yolo/data.yaml --weights weights/best.pt --split test
python src/infer_demo.py --weights weights/best.pt --source data/shwd_yolo/test/images --make-video
```

### Informe (PDF, formato NeurIPS)

```bash
# requiere un compilador LaTeX (p. ej. tectonic o pdflatex/latexmk)
tectonic report/neurips/main.tex          # -> report/neurips/main.pdf
# versión Markdown alternativa:
python report/build_pdf.py                 # -> report/REPORT.pdf
```

## Entregables (mapeo con la consigna)

1. **Modelo YOLO ajustado, decisiones documentadas** → `src/train.py` + §3.3 del informe (anchor-free, mosaic+close_mosaic, augmentación, imgsz 640).
2. **mAP@0.5 y mAP@0.5:0.95 por clase** → `src/evaluate.py` → `outputs/metrics_overall.json`.
3. **Robustez ante oclusión** (bajo/medio/alto) y **distancia** → `outputs/robustness_*`.
4. **Verificador de cumplimiento + demo** → `src/compliance_verifier.py`, `src/infer_demo.py` → `outputs/demo/`.
5. **Implicaciones éticas** → §6 del informe.

## Dataset y licencia

Dataset: **Safety-Helmet-Wearing-Dataset (SHWD)** — njvisionpower
(https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), formato Pascal VOC
convertido a YOLO. ~7 581 imágenes; clases `helmet` (con casco) y `head` (sin casco).
Es uno de los datasets recomendados por la consigna por incluir ejemplos negativos
explícitos de "sin casco". El backbone YOLOv8 es de Ultralytics (AGPL-3.0). Código de
este repositorio bajo MIT (ver `LICENSE`). Atribución del dataset según su licencia original.

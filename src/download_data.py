"""
Descarga el dataset de detección de cumplimiento de EPP desde Roboflow Universe.

Dataset: large-benchmark-datasets / logistics-sz9jr (version 2), formato YOLOv8.

La API key se lee de la variable de entorno ROBOFLOW_API_KEY para no exponerla
en el repositorio. Configúrala antes de ejecutar:

    Windows (PowerShell):  $env:ROBOFLOW_API_KEY = "tu_api_key"
    Linux / Colab:         export ROBOFLOW_API_KEY="tu_api_key"

Uso:
    python src/download_data.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

WORKSPACE = "large-benchmark-datasets"
PROJECT = "logistics-sz9jr"
VERSION = 2
FORMAT = "yolov8"


def main() -> str:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: define la variable de entorno ROBOFLOW_API_KEY con tu clave de Roboflow.\n"
            '  PowerShell:  $env:ROBOFLOW_API_KEY = "tu_api_key"\n'
            '  bash/Colab:  export ROBOFLOW_API_KEY="tu_api_key"'
        )

    from roboflow import Roboflow

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(DATA_DIR)  # Roboflow descarga en el cwd

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(FORMAT)

    print(f"\nDataset descargado en: {dataset.location}")
    print(f"data.yaml: {Path(dataset.location) / 'data.yaml'}")
    return dataset.location


if __name__ == "__main__":
    main()

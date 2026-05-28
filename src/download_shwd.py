"""
Descarga el Safety-Helmet-Wearing-Dataset (SHWD) desde Google Drive y lo extrae.

SHWD (njvisionpower) es uno de los datasets recomendados por la consigna y aporta
ejemplos negativos EXPLÍCITOS de "sin casco" (clase 'person' = cabeza descubierta),
fundamentales para un verificador de cumplimiento fiable. A diferencia de Roboflow,
NO requiere API key, por lo que el pipeline es reproducible sin credenciales.

Formato Pascal VOC (VOC2028). ~7581 imágenes. Clases: hat / person.

Uso:
    python src/download_shwd.py
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GDRIVE_ID = "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX"  # VOC2028.zip (~1.12 GB)
ZIP_PATH = DATA / "VOC2028.zip"
EXTRACT_DIR = DATA / "SHWD"
VOC_DIR = EXTRACT_DIR / "VOC2028"


def main() -> str:
    DATA.mkdir(parents=True, exist_ok=True)
    if VOC_DIR.exists() and (VOC_DIR / "JPEGImages").exists():
        print(f"SHWD ya extraído en {VOC_DIR}")
        return str(VOC_DIR)

    if not ZIP_PATH.exists():
        try:
            import gdown
        except ImportError:
            sys.exit("Falta 'gdown'. Instala con: pip install gdown")
        print("Descargando SHWD (VOC2028.zip ~1.12 GB) desde Google Drive...")
        gdown.download(id=GDRIVE_ID, output=str(ZIP_PATH), quiet=False)

    print(f"Extrayendo {ZIP_PATH} -> {EXTRACT_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)
    print(f"SHWD listo en {VOC_DIR}")
    return str(VOC_DIR)


if __name__ == "__main__":
    main()

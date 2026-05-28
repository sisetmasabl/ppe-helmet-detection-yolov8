"""
Convierte el Safety-Helmet-Wearing-Dataset (SHWD, formato Pascal VOC) al formato
YOLOv8 y construye los splits train/val/test a partir de los conjuntos
predefinidos en ImageSets/Main (sin fuga de datos).

Mapeo de clases (decisión documentada en el informe):
    hat    -> 0  helmet   (trabajador CON casco  -> CONFORME)
    person -> 1  head      (cabeza SIN casco      -> VIOLACIÓN "sin casco")
  (cualquier otra etiqueta, p. ej. 'dog', se descarta)

A diferencia del dataset Logistics anterior, SHWD incluye ejemplos negativos
EXPLÍCITOS de "sin casco" (clase 'person'=cabeza descubierta), que es justo lo
que la consigna pide para entrenar un verificador de cumplimiento fiable y para
el desglose por clase (con casco / sin casco).

Uso:
    python src/prepare_shwd.py --src data/SHWD/VOC2028 --dst data/shwd_yolo
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NAME2ID = {"hat": 0, "person": 1}
NEW_NAMES = ["helmet", "head"]  # 0=con casco (conforme), 1=sin casco (violación)


def convert_annotation(xml_path: Path):
    """Devuelve lista de líneas YOLO (clase xc yc w h normalizados)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    if w <= 0 or h <= 0:
        return None
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in NAME2ID:
            continue
        b = obj.find("bndbox")
        xmin = float(b.findtext("xmin"))
        ymin = float(b.findtext("ymin"))
        xmax = float(b.findtext("xmax"))
        ymax = float(b.findtext("ymax"))
        # clip a los límites de la imagen
        xmin = max(0.0, min(xmin, w)); xmax = max(0.0, min(xmax, w))
        ymin = max(0.0, min(ymin, h)); ymax = max(0.0, min(ymax, h))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{NAME2ID[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def build_split(src: Path, dst: Path, ids, split_name):
    img_out = dst / split_name / "images"
    lbl_out = dst / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    src_img = src / "JPEGImages"
    src_ann = src / "Annotations"
    n_img = 0
    n_helmet = 0
    n_head = 0
    n_empty = 0
    for stem in ids:
        xml_path = src_ann / f"{stem}.xml"
        jpg_path = src_img / f"{stem}.jpg"
        if not xml_path.exists() or not jpg_path.exists():
            continue
        lines = convert_annotation(xml_path)
        if lines is None:
            continue
        if not lines:
            n_empty += 1  # imagen sin objetos relevantes (fondo) -> se omite
            continue
        shutil.copy2(jpg_path, img_out / jpg_path.name)
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        n_img += 1
        for ln in lines:
            if ln.startswith("0 "):
                n_helmet += 1
            elif ln.startswith("1 "):
                n_head += 1
    print(f"  {split_name}: imgs={n_img}  helmet(con casco)={n_helmet}  "
          f"head(sin casco)={n_head}  omitidas_sin_objetos={n_empty}")
    return n_img


def read_ids(p: Path):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "SHWD" / "VOC2028"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "shwd_yolo"))
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    sets = src / "ImageSets" / "Main"
    print(f"Convirtiendo SHWD (VOC) -> YOLO en {dst}")

    for split, fname in (("train", "train.txt"), ("valid", "val.txt"), ("test", "test.txt")):
        ids = read_ids(sets / fname)
        build_split(src, dst, ids, split)

    data_yaml = dst / "data.yaml"
    lines = ["names:"]
    lines += [f"- {n}" for n in NEW_NAMES]
    lines += [
        f"nc: {len(NEW_NAMES)}",
        f"train: {(dst / 'train' / 'images').resolve()}",
        f"val: {(dst / 'valid' / 'images').resolve()}",
        f"test: {(dst / 'test' / 'images').resolve()}",
        f"path: {dst.resolve()}",
    ]
    data_yaml.write_text("\n".join(lines))
    print(f"\ndata.yaml escrito en {data_yaml}")
    print(data_yaml.read_text())


if __name__ == "__main__":
    main()

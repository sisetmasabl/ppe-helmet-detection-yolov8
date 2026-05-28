"""
Construye un subconjunto ENFOCADO EN LA TAREA a partir del dataset Logistics-2
(20 clases, ~95k imágenes) para hacer factible el entrenamiento de 100 épocas en
una GPU de 8 GB, sin perder el foco en el cumplimiento de EPP (casco).

Decisiones (documentadas en el informe):
  - Se conservan solo las clases relevantes para EPP y se remapean:
        0: person       (orig 10)
        1: helmet       (orig 7)
        2: safety vest  (orig 13)
        3: gloves       (orig 6)
  - Se seleccionan imágenes que contienen 'person' o 'helmet', priorizando
    las que contienen 'helmet' (clase clave, más escasa) y las más ricas en
    instancias, para maximizar la señal de casco a corta y larga distancia.
  - Se respeta la división original train/valid/test (sin fuga de datos).

Uso:
    python src/prepare_subset.py --src data/Logistics-2 \
        --dst data/ppe_subset --train 5000 --valid 1200 --test 2000
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD2NEW = {10: 0, 7: 1, 13: 2, 6: 3}
NEW_NAMES = ["person", "helmet", "safety vest", "gloves"]
HELMET_OLD = 7
PERSON_OLD = 10
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def scan_split(src_split: Path):
    """Devuelve lista de (stem, has_helmet, n_target) para imágenes con person/helmet."""
    img_dir = src_split / "images"
    lbl_dir = src_split / "labels"
    items = []
    n_total = 0
    for lf in lbl_dir.glob("*.txt"):
        n_total += 1
        has_helmet = False
        has_person = False
        n_target = 0
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c in OLD2NEW:
                n_target += 1
                if c == HELMET_OLD:
                    has_helmet = True
                if c == PERSON_OLD:
                    has_person = True
        if has_helmet or has_person:
            items.append((lf.stem, has_helmet, n_target))
    return items, n_total


def select(items, cap):
    # prioridad: con casco primero, luego por nº de instancias relevantes
    items_sorted = sorted(items, key=lambda t: (not t[1], -t[2]))
    return items_sorted[:cap]


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def write_label(src_lbl: Path, dst_lbl: Path):
    out = []
    for line in src_lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        if c in OLD2NEW:
            out.append(" ".join([str(OLD2NEW[c])] + p[1:5]))
    dst_lbl.write_text("\n".join(out))


def build_split(src_split: Path, dst_split: Path, cap: int):
    items, n_total = scan_split(src_split)
    n_helmet_imgs = sum(1 for _, hh, _ in items if hh)
    chosen = select(items, cap)
    n_helmet_chosen = sum(1 for _, hh, _ in chosen if hh)

    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    src_img = src_split / "images"
    src_lbl = src_split / "labels"
    n_copied = 0
    for stem, _, _ in chosen:
        ip = find_image(src_img, stem)
        if ip is None:
            continue
        shutil.copy2(ip, dst_split / "images" / ip.name)
        write_label(src_lbl / (stem + ".txt"), dst_split / "labels" / (stem + ".txt"))
        n_copied += 1
    print(f"  {src_split.name}: total={n_total}  con_person/helmet={len(items)}  "
          f"con_helmet={n_helmet_imgs}  elegidas={n_copied} (con_helmet={n_helmet_chosen})")
    return n_copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "Logistics-2"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "ppe_subset"))
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--valid", type=int, default=1200)
    ap.add_argument("--test", type=int, default=2000)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    print(f"Construyendo subconjunto EPP en {dst}")
    caps = {"train": args.train, "valid": args.valid, "test": args.test}
    for split in ("train", "valid", "test"):
        build_split(src / split, dst / split, caps[split])

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
    print((data_yaml).read_text())


if __name__ == "__main__":
    main()

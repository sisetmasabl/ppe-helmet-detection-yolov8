"""
Verificador de cumplimiento de EPP basado en reglas (módulo de posprocesamiento).

Soporta DOS esquemas de clases, eligiendo automáticamente según el dataset:

1) CLASIFICACIÓN DIRECTA (dataset SHWD: clases 'helmet' y 'head').
   El detector ya distingue cabeza CON casco ('helmet') de cabeza SIN casco
   ('head'). Regla:
        - helmet -> trabajador CONFORME
        - head   -> VIOLACIÓN ("sin casco")
        - tasa_cumplimiento = #helmet / (#helmet + #head)

2) ASOCIACIÓN ESPACIAL (dataset con 'person' + 'helmet', sin clase "sin casco").
   Un trabajador (person) cumple si tiene un casco (helmet) asociado en la región
   superior de su bounding box; en caso contrario es VIOLACIÓN.
        - tasa_cumplimiento = #personas_con_casco / #personas_totales

En ambos casos se dibuja: conforme (verde), violación "sin casco" (rojo, grueso),
cascos sueltos (cian) y un encabezado con la tasa de cumplimiento por fotograma.
Esta lógica maneja directamente el reto de larga distancia: una cabeza/casco
lejano (objeto pequeño) mal detectado afecta la tasa, lo que es relevante para el
análisis de falsos negativos.

Uso:
    from compliance_verifier import ComplianceVerifier
    cv = ComplianceVerifier(class_names)
    res = cv.process_frame(frame_bgr, boxes_xyxy, cls_ids, confs)
    res.annotated, res.compliance_rate, res.counts, res.violations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

PERSON_KEYS = ["person", "persona", "worker", "trabajador"]
HELMET_KEYS = ["helmet", "hardhat", "hard-hat", "hat", "casco"]
NOHELMET_KEYS = ["head", "no-helmet", "nohelmet", "no_helmet", "sin casco", "without"]
VEST_KEYS = ["vest", "chaleco"]

# Colores BGR
COLOR_OK = (60, 180, 75)         # verde  -> conforme (con casco)
COLOR_VIOLATION = (60, 60, 230)  # rojo   -> sin casco
COLOR_HELMET = (230, 200, 60)    # cian   -> casco suelto (modo asociación)
COLOR_OTHER = (170, 170, 170)    # gris   -> otras clases


def _match_class(names: Sequence[str], keys) -> List[int]:
    out = []
    for i, n in enumerate(names):
        low = str(n).lower().strip()
        if any(k in low for k in keys):
            out.append(i)
    return out


@dataclass
class FrameResult:
    annotated: np.ndarray
    compliance_rate: Optional[float]            # None si no hay trabajadores
    counts: Dict[str, int]                       # persons, compliant, violation, helmets
    violations: List[dict] = field(default_factory=list)


class ComplianceVerifier:
    def __init__(
        self,
        class_names: Sequence[str],
        conf_threshold: float = 0.25,
        person_ids: Optional[List[int]] = None,
        helmet_ids: Optional[List[int]] = None,
        nohelmet_ids: Optional[List[int]] = None,
        upper_frac: float = 0.5,     # región superior de la persona donde puede ir el casco
        top_margin: float = 0.15,    # margen por encima de la cabeza (fracción de la altura)
        x_margin: float = 0.15,      # margen horizontal (fracción del ancho)
    ):
        self.class_names = list(class_names)
        self.conf_threshold = conf_threshold
        self.nohelmet_ids = nohelmet_ids if nohelmet_ids is not None else _match_class(self.class_names, NOHELMET_KEYS)
        # 'helmet'/'hat' es casco; excluir ids que ya son "sin casco" (p. ej. una clase
        # llamada literalmente 'no-helmet' contiene la subcadena 'helmet').
        _hel = helmet_ids if helmet_ids is not None else _match_class(self.class_names, HELMET_KEYS)
        self.helmet_ids = [i for i in _hel if i not in self.nohelmet_ids]
        # 'person' solo en modo asociación; excluir ids que ya son nohelmet/helmet
        ppl = person_ids if person_ids is not None else _match_class(self.class_names, PERSON_KEYS)
        self.person_ids = [i for i in ppl if i not in self.nohelmet_ids and i not in self.helmet_ids]
        self.vest_ids = _match_class(self.class_names, VEST_KEYS)
        self.upper_frac = upper_frac
        self.top_margin = top_margin
        self.x_margin = x_margin
        # Modo: si hay clase explícita "sin casco" -> clasificación directa
        self.direct_mode = len(self.nohelmet_ids) > 0

    # ---------- modo asociación (person + helmet) ----------
    def _helmet_matches_person(self, helmet_box, person_box) -> bool:
        hx = (helmet_box[0] + helmet_box[2]) / 2.0
        hy = (helmet_box[1] + helmet_box[3]) / 2.0
        px1, py1, px2, py2 = person_box
        pw = max(px2 - px1, 1.0); ph = max(py2 - py1, 1.0)
        x_ok = (px1 - self.x_margin * pw) <= hx <= (px2 + self.x_margin * pw)
        y_ok = (py1 - self.top_margin * ph) <= hy <= (py1 + self.upper_frac * ph)
        return bool(x_ok and y_ok)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        cls_ids: Sequence[int],
        confs: Sequence[float],
    ) -> FrameResult:
        if self.direct_mode:
            return self._process_direct(frame_bgr, boxes_xyxy, cls_ids, confs)
        return self._process_association(frame_bgr, boxes_xyxy, cls_ids, confs)

    # ---------- CLASIFICACIÓN DIRECTA (SHWD) ----------
    def _process_direct(self, frame_bgr, boxes_xyxy, cls_ids, confs) -> FrameResult:
        img = frame_bgr.copy()
        helmets, heads, others = [], [], []
        for box, cid, conf in zip(boxes_xyxy, cls_ids, confs):
            if conf < self.conf_threshold:
                continue
            cid = int(cid); box = [float(v) for v in box]
            if cid in self.helmet_ids:
                helmets.append({"box": box, "conf": float(conf)})
            elif cid in self.nohelmet_ids:
                heads.append({"box": box, "conf": float(conf)})
            else:
                others.append({"box": box, "conf": float(conf)})

        violations = [{"box": [int(v) for v in h["box"]], "conf": h["conf"]} for h in heads]
        for o in others:
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OTHER, 1)
        for hm in helmets:
            x1, y1, x2, y2 = [int(v) for v in hm["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OK, 2)
            self._label(img, x1, y1, f"OK casco {hm['conf']:.2f}", COLOR_OK)
        for hd in heads:
            x1, y1, x2, y2 = [int(v) for v in hd["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_VIOLATION, 3)
            self._label(img, x1, y1, f"SIN CASCO {hd['conf']:.2f}", COLOR_VIOLATION)

        n_ok = len(helmets); n_viol = len(heads); n_workers = n_ok + n_viol
        rate = (n_ok / n_workers) if n_workers > 0 else None
        counts = {"persons": n_workers, "compliant": n_ok, "violation": n_viol, "helmets": n_ok}
        self._header(img, rate, counts)
        return FrameResult(annotated=img, compliance_rate=rate, counts=counts, violations=violations)

    # ---------- ASOCIACIÓN ESPACIAL (person + helmet) ----------
    def _process_association(self, frame_bgr, boxes_xyxy, cls_ids, confs) -> FrameResult:
        img = frame_bgr.copy()
        persons, helmets, others = [], [], []
        for box, cid, conf in zip(boxes_xyxy, cls_ids, confs):
            if conf < self.conf_threshold:
                continue
            cid = int(cid); box = [float(v) for v in box]
            if cid in self.person_ids:
                persons.append({"box": box, "conf": float(conf)})
            elif cid in self.helmet_ids:
                helmets.append({"box": box, "conf": float(conf)})
            else:
                others.append({"box": box, "conf": float(conf), "cid": cid})

        helmet_used = [False] * len(helmets)
        violations = []
        n_compliant = 0
        for pr in persons:
            matched = False
            for j, hm in enumerate(helmets):
                if helmet_used[j]:
                    continue
                if self._helmet_matches_person(hm["box"], pr["box"]):
                    helmet_used[j] = True
                    matched = True
                    break
            pr["compliant"] = matched
            if matched:
                n_compliant += 1
            else:
                violations.append({"box": [int(v) for v in pr["box"]], "conf": pr["conf"]})

        for o in others:
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OTHER, 1)
        for hm in helmets:
            x1, y1, x2, y2 = [int(v) for v in hm["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_HELMET, 2)
        for pr in persons:
            x1, y1, x2, y2 = [int(v) for v in pr["box"]]
            if pr["compliant"]:
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OK, 2)
                self._label(img, x1, y1, f"OK casco {pr['conf']:.2f}", COLOR_OK)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_VIOLATION, 3)
                self._label(img, x1, y1, f"SIN CASCO {pr['conf']:.2f}", COLOR_VIOLATION)

        n_persons = len(persons)
        rate = (n_compliant / n_persons) if n_persons > 0 else None
        counts = {"persons": n_persons, "compliant": n_compliant,
                  "violation": n_persons - n_compliant, "helmets": len(helmets)}
        self._header(img, rate, counts)
        return FrameResult(annotated=img, compliance_rate=rate, counts=counts, violations=violations)

    @staticmethod
    def _label(img, x, y, text, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        y_top = max(0, y - th - 4)
        cv2.rectangle(img, (x, y_top), (x + tw + 4, y_top + th + 4), color, -1)
        cv2.putText(img, text, (x + 2, y_top + th), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _header(img, rate, counts):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        if rate is None:
            txt = "Cumplimiento: N/A (sin trabajadores detectados)"
            col = (200, 200, 200)
        else:
            pct = rate * 100
            txt = (f"Cumplimiento casco: {pct:.0f}%  "
                   f"(trabajadores={counts['persons']} OK={counts['compliant']} "
                   f"SIN_CASCO={counts['violation']})")
            col = COLOR_OK if counts["violation"] == 0 else COLOR_VIOLATION
        cv2.putText(img, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)

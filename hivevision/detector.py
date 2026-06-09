"""Tile detector inference: a trained YOLO-pose checkpoint -> per-tile detections.

The learned half of the pipeline (plan.md Phase 3). Each detection is one Hive tile:
its **class** (one of the 16 codes in ``hivevision.pieces``) and its **icon-centre
keypoint** in image pixels. The icon centre is coplanar with the table, so feeding the
detected centres straight into ``hivevision.geometry.recover_lattice`` is Phase 4.

``ultralytics`` is imported lazily (it lives in the ``yolo`` dependency group), so importing
this module is cheap and does not require the detector deps until you actually load a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hivevision.pieces import CLASSES


@dataclass(frozen=True)
class Detection:
    """One detected tile."""

    code: str  # class code, e.g. "wQ"
    cls: int  # class index into pieces.CLASSES
    conf: float
    xy: tuple[float, float]  # icon-centre keypoint, image pixels
    box: tuple[float, float, float, float]  # xyxy, image pixels


class TileDetector:
    """Wrap an Ultralytics YOLO-pose checkpoint for tile detection."""

    def __init__(self, ckpt: str | Path, device: str | None = None):
        from ultralytics import YOLO  # lazy: only needed when a model is loaded

        self.model = YOLO(str(ckpt))
        self.device = device

    def detect(
        self,
        image: str | Path | np.ndarray,
        conf: float = 0.25,
        imgsz: int = 1280,
    ) -> list[Detection]:
        """Detect tiles in one image (path, or an RGB uint8 array).

        Returns detections sorted by descending confidence.
        """
        if isinstance(image, np.ndarray):
            source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ultralytics np source is BGR
        else:
            source = str(image)

        result = self.model.predict(
            source, conf=conf, imgsz=imgsz, device=self.device, verbose=False
        )[0]
        if result.keypoints is None or result.boxes is None:
            return []

        kpts = result.keypoints.xy.cpu().numpy()  # (N, 1, 2)
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = result.boxes.conf.cpu().numpy()  # (N,)
        clss = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        dets = [
            Detection(
                code=CLASSES[c],
                cls=int(c),
                conf=float(cf),
                xy=(float(kp[0, 0]), float(kp[0, 1])),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            )
            for kp, b, cf, c in zip(kpts, boxes, confs, clss, strict=True)
        ]
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets

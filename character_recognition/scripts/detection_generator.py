"""
text_box_detector.py
--------------------

Lightweight wrapper around the Ultralytics YOLOv8 detection model that exposes a simple
Python API for detecting *text* bounding‑boxes in an image.

Typical usage
-------------

```python
from text_box_detector import TextBoxDetector

detector = TextBoxDetector(weights="weights/text_yolo8.pt", class_names=["text"])
detections = detector("page.png")

for det in detections:
    print(det)
    # {'bbox': (x1, y1, x2, y2), 'confidence': 0.93, 'class_id': 0, 'class_name': 'text'}
```
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Sequence, Optional, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ultralytics package is required: `pip install ultralytics`."
    ) from exc


class TextBoxDetector:
    """Detect text bounding‑boxes in images using a YOLOv8 model.

    Parameters
    ----------
    weights:
        Path to a trained YOLOv8 `.pt` weights file (or an Ultralytics model name).
    device:
        Torch device string (e.g. ``'cpu'`` or ``'cuda:0'``).  ``None`` lets
        Ultralytics decide.
    class_names:
        Optional explicit list or tuple of class names **ordered by the class id
        used during training**.  If *None*, the class names embedded in the model
        will be used when available; otherwise ``'unknown_{id}'`` placeholders are
        generated.
    conf:
        Confidence threshold for post‑processing (0–1).
    iou:
        Non‑max suppression IoU threshold (0–1).
    """

    def __init__(
        self,
        *,
        weights: Union[str, Path],
        device: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.model = YOLO(str(weights))
        if device:
            self.model.to(device)

        # confidence / iou thresholds for predict()
        self._conf = conf
        self._iou = iou

        # fallback to model's metadata if explicit names not supplied
        model_names = getattr(self.model, "names", None) or {}
        self.class_names = dict(enumerate(class_names)) if class_names else model_names

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def __call__(
        self,
        image: Union[str, Path, np.ndarray],
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        agnostic_nms: bool = False,
    ) -> List[Dict]:
        """Detect text boxes in *image* and return them as a list of dicts.

        Each returned dict has the form::

            {
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'class_id': int,
                'class_name': str,
            }

        Notes
        -----
        *   The bounding‑box coordinates are in pixel space **relative to the
            *original* image resolution** (not the resized image fed to YOLO).
        *   The ``class_name`` is looked‑up from the ``class_names`` mapping.
        """
        return self.detect(
            image,
            conf=conf if conf is not None else self._conf,
            iou=iou if iou is not None else self._iou,
            agnostic_nms=agnostic_nms,
        )

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        *,
        conf: float,
        iou: float,
        agnostic_nms: bool = False,
    ) -> List[Dict]:
        # ensure we have a numpy array to keep aspect ratio info
        img_array = self._load_image(image)
        h, w = img_array.shape[:2]

        # predictions
        results = self.model.predict(
            source=img_array,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )

        if not results:
            return []

        # Ultralytics predict() returns a list, one item per image
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = boxes.conf.cpu().numpy()  # (N,)
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

        detections: List[Dict] = []
        for (x1, y1, x2, y2), score, cid in zip(xyxy, confs, cls_ids):
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": float(score),
                    "class_id": int(cid),
                    "class_name": self.class_names.get(
                        int(cid), f"unknown_{cid}"
                    ),
                }
            )

        return detections

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_image(data: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Read *data* into a BGR ndarray (OpenCV convention)."""
        if isinstance(data, np.ndarray):
            return data
        # treat as a path
        data = Path(data)
        if not data.exists():
            raise FileNotFoundError(data)
        img = cv2.imread(str(data))  # BGR
        if img is None:
            raise ValueError(f"Unable to read image at {data}")
        return img


# ---------------------------------------------------------------------- #
# functional helper
# ---------------------------------------------------------------------- #

def detect_text_boxes(
    image: Union[str, Path, np.ndarray],
    *,
    weights: Union[str, Path],
    device: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
) -> List[Dict]:
    """Convenience wrapper around :class:`TextBoxDetector`.

    Parameters
    ----------
    image:
        Image path or ndarray.
    weights, device, class_names, conf, iou:
        Passed straight to :class:`TextBoxDetector`.
    """
    detector = TextBoxDetector(
        weights=weights,
        device=device,
        class_names=class_names,
        conf=conf,
        iou=iou,
    )
    return detector(image)

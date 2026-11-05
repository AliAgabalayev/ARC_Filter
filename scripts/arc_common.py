"""Shared helpers for the Fashionpedia ARC benchmark.

Paths (relative to the repository root), the mixed-noise model, a thin RF-DETR
wrapper and the COCO mask evaluation used by ``arc_full_valid.py``.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "rfdetr_fashionpedia_b4.pth"
VALID_DIR = ROOT / "data" / "fashionpedia_2020_v1_split" / "valid"
VALID_ANN = VALID_DIR / "_annotations.coco.json"

RESOLUTION = 768
MAX_DETS = [1, 10, 100]
SEED = 42
SIGMA = 0.10
SP_FRACTION = 0.01


def to_uint8(rgb01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb01, 0, 1) * 255).astype(np.uint8)


def add_mixed_noise(rgb01: np.ndarray, image_id: int, seed: int = SEED) -> np.ndarray:
    """Gaussian (sigma=0.10, clipped) then salt-and-pepper on 1% of pixels (all channels).

    The RNG is derived deterministically from (seed, image_id).
    """
    rng = np.random.default_rng([seed, int(image_id)])
    noisy = np.clip(rgb01 + rng.normal(0.0, SIGMA, rgb01.shape), 0.0, 1.0)
    h, w = noisy.shape[:2]
    sp = rng.random((h, w)) < SP_FRACTION          # ~1% of pixels
    salt = rng.random((h, w)) < 0.5                # white/black equiprobable
    noisy[sp & salt] = 1.0
    noisy[sp & ~salt] = 0.0
    return noisy.astype(np.float32)


def load_model():
    """Load the fine-tuned RF-DETR Seg Small checkpoint at the evaluation resolution."""
    from rfdetr import RFDETRSegSmall  # imported lazily: only needed for inference
    return RFDETRSegSmall.from_checkpoint(str(CHECKPOINT), resolution=RESOLUTION)


def check_class_mapping(model, coco: COCO) -> None:
    """Model class_id (0-based list index) -> COCO category_id == class_id + 1."""
    cats = sorted(coco.dataset["categories"], key=lambda c: c["id"])
    names = list(model.class_names)
    assert [c["name"] for c in cats] == names, "model class names differ from annotation categories"
    assert [c["id"] for c in cats] == list(range(1, len(names) + 1)), "category ids not 1..N"


def predict(model, img_u8: np.ndarray, threshold: float) -> dict:
    det = model.predict(Image.fromarray(img_u8), threshold=threshold)
    h, w = img_u8.shape[:2]
    n = len(det)
    masks = det.mask if n else np.zeros((0, h, w), bool)
    assert masks.shape[1:] == (h, w), f"mask size {masks.shape[1:]} != image {(h, w)}"
    return {
        "masks": masks.astype(bool),
        "class_ids": det.class_id.astype(int) if n else np.zeros(0, int),
        "scores": det.confidence.astype(float) if n else np.zeros(0, float),
    }


def to_rle(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def evaluate(coco_gt: COCO, dets: list[dict], img_ids: list[int], log_path: Path) -> dict:
    """COCO mask evaluation (maxDets=[1,10,100]); per-class AP is None where a class has no GT."""
    cat_ids = sorted(coco_gt.getCatIds())
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": None,
                "per_class": {c: None for c in cat_ids}}
    dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, dt, "segm")
    ev.params.imgIds = img_ids
    ev.params.maxDets = MAX_DETS
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    log_path.write_text(buf.getvalue())
    st = ev.stats
    na = lambda x: None if x < 0 else float(x)  # noqa: E731
    prec = ev.eval["precision"]  # [T,R,K,A,M]; area 'all' = 0, maxDets=100 = last
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, len(MAX_DETS) - 1]
        p = p[p > -1]
        per_class[cid] = float(p.mean()) if p.size else None
    return {"AP": na(st[0]), "AP50": na(st[1]), "AP75": na(st[2]), "AP_small": na(st[3]),
            "per_class": per_class}

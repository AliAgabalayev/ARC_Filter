"""Compare V-channel filters (none / median / bilateral / ARC) for a fixed RF-DETR Seg Small model
on the Fashionpedia validation split, with clean and mixed-noise inputs (16 dataset passes).

Expected inputs, relative to the repository root:
  checkpoints/rfdetr_fashionpedia_b4.pth                      fine-tuned RF-DETR Seg Small checkpoint
  data/fashionpedia_2020_v1_split/valid/                       validation images
  data/fashionpedia_2020_v1_split/valid/_annotations.coco.json COCO-format annotations

Stages (all resumable):
  run    sharded predictions   -> <out>/preds/chunk_XXX.json.gz
  eval   COCO segm evaluation  -> <out>/evals/<pass>.json
  report summary tables        -> <out>/{results_table.md, metrics.json, per_class.csv, timing.json}

Extra dependencies: see requirements-fashionpedia.txt.
"""
import argparse, contextlib, csv, gzip, io, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from arc_common import (SIGMA, SP_FRACTION, VALID_ANN, VALID_DIR, add_mixed_noise, predict,  # noqa: E402
                        to_rle, to_uint8)
from arc_filter import ARC_filter  # noqa: E402

METHODS = ["none", "median", "bilateral", "arc"]
SEEDS = [42, 43, 44]
SOURCES = ["clean"] + [f"noisy{s}" for s in SEEDS]
PASSES = [f"{src}_{m}" for src in SOURCES for m in METHODS]  # 16 dataset passes
SCORE_THR = 0.05
RESOLUTION = 768
MAX_DETS = [1, 10, 100]
CHUNK = 50
BILATERAL = dict(d=5, sigmaColor=1 / 3, sigmaSpace=1)
MEDIAN_K = 3
CONFIG = {
    "checkpoint": "checkpoints/rfdetr_fashionpedia_b4.pth", "resolution": RESOLUTION,
    "dataset": "data/fashionpedia_2020_v1_split/valid (all images)",
    "methods": METHODS, "passes": PASSES,
    "score_threshold": SCORE_THR, "mask_threshold": "rfdetr default (identical for all passes)",
    "evaluator": {"tool": "pycocotools COCOeval", "iouType": "segm", "maxDets": MAX_DETS},
    "class_map": "category_id = model class_id + 1 (verified by name at start)",
    "noise": {"seeds": SEEDS, "rng": "default_rng([seed, image_id])", "gaussian_sigma": SIGMA,
              "salt_pepper_fraction": SP_FRACTION,
              "same_noisy_array_for_all_methods": True, "noisy_images_written_to_disk": False},
    "filters": {"applied_to": "V channel of HSV (float32 [0,1]) at original size, then clip [0,1] -> RGB -> uint8 -> model",
                "none": "no HSV round trip (identity round trip verified lossless in uint8)",
                "median": {"cv2.medianBlur": MEDIAN_K, "border": "replicate (OpenCV)"},
                "bilateral": {"cv2.bilateralFilter": BILATERAL, "border": "reflect101 (OpenCV)"},
                "arc": {"impl": "arc_filter.ARC_filter", "window": "3x3", "border": "symmetric pad"}},
    "std": "sample std (ddof=1) over the 3 noise seeds",
    "timing": "filter_s = HSV conversion + V filter + RGB back + uint8 (noise generation excluded, reported separately); "
              "infer_s = model.predict incl. its preprocessing and mask upsampling",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def filter_rgb(method: str, rgb01: np.ndarray) -> np.ndarray:
    if method == "none":
        return to_uint8(rgb01)
    hsv = cv2.cvtColor(rgb01.astype(np.float32), cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    if method == "median":
        v = cv2.medianBlur(v, MEDIAN_K)
    elif method == "bilateral":
        v = cv2.bilateralFilter(v, **BILATERAL)
    elif method == "arc":
        v = ARC_filter(v.astype(np.float64)).astype(np.float32)
    else:
        raise ValueError(method)
    hsv[:, :, 2] = np.clip(v, 0.0, 1.0)
    return to_uint8(np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0))


def load_images():
    ann = json.load(open(VALID_ANN))
    return sorted(ann["images"], key=lambda r: r["id"])


# ------------------------------------------------------------------ run stage
def process_chunk(model, recs, chunk_path: Path):
    dets = {p: [] for p in PASSES}
    tim = {p: {"filter_s": 0.0, "infer_s": 0.0, "n": 0} for p in PASSES}
    noise_s = 0.0
    for rec in recs:
        iid = rec["id"]
        rgb = np.asarray(Image.open(VALID_DIR / rec["file_name"]).convert("RGB"), np.float32) / 255.0
        assert rgb.shape[:2] == (rec["height"], rec["width"])
        for src in SOURCES:
            if src == "clean":
                arr = rgb
            else:
                t = time.perf_counter(); arr = add_mixed_noise(rgb, iid, int(src[5:])); noise_s += time.perf_counter() - t
            for m in METHODS:  # same `arr` for every method
                p = f"{src}_{m}"
                t = time.perf_counter(); u8 = filter_rgb(m, arr); tim[p]["filter_s"] += time.perf_counter() - t
                t = time.perf_counter(); pr = predict(model, u8, SCORE_THR); tim[p]["infer_s"] += time.perf_counter() - t
                tim[p]["n"] += 1
                for mk, c, s in zip(pr["masks"], pr["class_ids"], pr["scores"]):
                    dets[p].append({"image_id": iid, "category_id": int(c) + 1, "score": round(float(s), 5),
                                    "segmentation": to_rle(mk)})
    tmp = chunk_path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump({"images": [r["id"] for r in recs], "dets": dets, "timing": tim, "noise_gen_s": noise_s}, f)
    os.replace(tmp, chunk_path)


def run_stage(out: Path, limit):
    import torch
    from arc_common import check_class_mapping, load_model
    from pycocotools.coco import COCO
    recs = load_images()[:limit] if limit else load_images()
    log(f"images: {len(recs)}  passes: {len(PASSES)}  chunks of {CHUNK}")
    model = load_model()  # once
    with contextlib.redirect_stdout(io.StringIO()):
        check_class_mapping(model, COCO(str(VALID_ANN)))
    log("class mapping verified (category_id = class_id + 1)")
    (out / "preds").mkdir(parents=True, exist_ok=True)
    chunks = [recs[i:i + CHUNK] for i in range(0, len(recs), CHUNK)]
    t0, done_imgs, todo = time.time(), 0, 0
    for k, ch in enumerate(chunks):
        path = out / "preds" / f"chunk_{k:03d}.json.gz"
        if path.exists():
            try:
                with gzip.open(path, "rt") as f:
                    ok = json.load(f)["images"] == [r["id"] for r in ch]
                if ok:
                    log(f"chunk {k + 1}/{len(chunks)} exists, skipping"); continue
            except Exception:
                log(f"chunk {k} unreadable, recomputing")
        for attempt in range(1, 4):
            try:
                tc = time.time(); process_chunk(model, ch, path)
                done_imgs += len(ch)
                el = time.time() - t0
                rest = sum(len(c) for c in chunks[k + 1:])
                log(f"chunk {k + 1}/{len(chunks)} done in {time.time() - tc:.0f}s | "
                    f"ETA {el / done_imgs * rest / 60:.0f} min for {rest} imgs")
                break
            except Exception as e:
                log(f"chunk {k} attempt {attempt} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                if attempt == 3:
                    raise
                time.sleep(5)
    log("run stage complete")


# ------------------------------------------------------------------ eval stage
def eval_worker(args):
    out, pass_name, limit = args
    from pycocotools.coco import COCO
    from arc_common import evaluate
    res_path = out / "evals" / f"{pass_name}.json"
    if res_path.exists():
        return pass_name, "cached"
    dets, ids = [], []
    for sh in sorted((out / "preds").glob("chunk_*.json.gz")):
        with gzip.open(sh, "rt") as f:
            d = json.load(f)
        ids += d["images"]; dets += d["dets"][pass_name]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(VALID_ANN))
    r = evaluate(coco, dets, ids, out / "evals" / f"{pass_name}.log")
    r["per_class"] = {str(k): v for k, v in r["per_class"].items()}
    r["n_images"], r["n_dets"] = len(ids), len(dets)
    res_path.write_text(json.dumps(r))
    return pass_name, "done"


def eval_stage(out: Path, limit, workers=3):
    (out / "evals").mkdir(exist_ok=True)
    log(f"eval stage: {len(PASSES)} passes, {workers} workers")
    with ProcessPoolExecutor(workers) as ex:
        for name, st in ex.map(eval_worker, [(out, p, limit) for p in PASSES]):
            log(f"eval {name}: {st}")


# ------------------------------------------------------------------ report stage
def fmt(x, d=1):
    return "N/A" if x is None else f"{100 * x:.{d}f}"


def ms(x, s=None):
    return "N/A" if x is None else (f"{100 * x:.1f}" + (f" ± {100 * s:.1f}" if s is not None else ""))


def report_stage(out: Path):
    ev = {p: json.load(open(out / "evals" / f"{p}.json")) for p in PASSES}
    ann = json.load(open(VALID_ANN))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    ids = set()
    for sh in sorted((out / "preds").glob("chunk_*.json.gz")):
        with gzip.open(sh, "rt") as f:
            ids |= set(json.load(f)["images"])
    gt_n = {c: 0 for c in cats}
    for a in ann["annotations"]:
        if a["image_id"] in ids:
            gt_n[a["category_id"]] += 1
    timing = {p: {"filter_s": 0.0, "infer_s": 0.0, "n": 0} for p in PASSES}
    noise_s = 0.0
    for sh in sorted((out / "preds").glob("chunk_*.json.gz")):
        with gzip.open(sh, "rt") as f:
            d = json.load(f)
        noise_s += d["noise_gen_s"]
        for p, t in d["timing"].items():
            for k in t:
                timing[p][k] += t[k]

    KEYS = ["AP", "AP50", "AP75", "AP_small"]
    NAMES = {"AP": "AP@[.50:.95]", "AP50": "AP50", "AP75": "AP75", "AP_small": "AP_small"}
    seedp = lambda m: [f"noisy{s}_{m}" for s in SEEDS]  # noqa: E731
    val = lambda p, k: ev[p][k]  # noqa: E731

    def mean_std(vals):
        if any(v is None for v in vals):
            return None, None
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    M = {"clean": {m: {k: val(f"clean_{m}", k) for k in KEYS} for m in METHODS},
         "noisy_seeds": {m: {s: {k: val(f"noisy{s}_{m}", k) for k in KEYS} for s in SEEDS} for m in METHODS},
         "noisy_mean_std": {m: {k: mean_std([val(p, k) for p in seedp(m)]) for k in KEYS} for m in METHODS},
         "delta_pp_clean": {}, "delta_pp_noisy_paired": {}}
    for m in METHODS[1:]:
        M["delta_pp_clean"][m] = {k: None if val(f"clean_{m}", k) is None or val("clean_none", k) is None
                                  else 100 * (val(f"clean_{m}", k) - val("clean_none", k)) for k in KEYS}
        M["delta_pp_noisy_paired"][m] = {k: mean_std([None if val(f"noisy{s}_{m}", k) is None or val(f"noisy{s}_none", k) is None
                                                      else 100 * (val(f"noisy{s}_{m}", k) - val(f"noisy{s}_none", k)) for s in SEEDS])
                                         for k in KEYS}

    pc = {p: {int(c): a for c, a in ev[p]["per_class"].items()} for p in PASSES}
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class", "gt_n"] + [f"{p}" for p in PASSES]
                   + [f"noisy_mean_{m}" for m in METHODS] + [f"noisy_std_{m}" for m in METHODS])
        for c in sorted(cats):
            ms_ = {m: mean_std([pc[p][c] for p in seedp(m)]) for m in METHODS}
            w.writerow([c, cats[c], gt_n[c]] + ["N/A" if pc[p][c] is None else round(100 * pc[p][c], 2) for p in PASSES]
                       + ["N/A" if ms_[m][0] is None else round(100 * ms_[m][0], 2) for m in METHODS]
                       + ["N/A" if ms_[m][1] is None else round(100 * ms_[m][1], 2) for m in METHODS])

    T = {m: {"filter_ms_per_img": 1000 * sum(timing[p]["filter_s"] for p in PASSES if p.endswith("_" + m)) /
                                  sum(timing[p]["n"] for p in PASSES if p.endswith("_" + m)),
             "infer_ms_per_img": 1000 * sum(timing[p]["infer_s"] for p in PASSES if p.endswith("_" + m)) /
                                 sum(timing[p]["n"] for p in PASSES if p.endswith("_" + m))} for m in METHODS}
    n_img = len(ids)
    T["noise_gen_ms_per_noisy_image"] = 1000 * noise_s / (n_img * len(SEEDS))
    (out / "timing.json").write_text(json.dumps({"per_pass": timing, "summary": T}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"n_images": n_img, "metrics": M, "gt_instances": {cats[c]: n for c, n in gt_n.items()},
                                                  "dets_per_pass": {p: ev[p]["n_dets"] for p in PASSES}}, indent=1))

    L = [f"# ARC full valid comparison ({n_img} images)", "",
         "Mask AP (COCO segm), %, score threshold 0.05, maxDets=[1,10,100], resolution 768. "
         "Filters act on the HSV V channel at original size. Differences are percentage points (pp).", "",
         "## Clean input", "", "| Method | " + " | ".join(NAMES[k] for k in KEYS) + " |", "|---|" + "---|" * 4]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(fmt(M["clean"][m][k]) for k in KEYS) + " |")
    L += ["", "Δ vs no-filter (pp):", "", "| Method | " + " | ".join(NAMES[k] for k in KEYS) + " |", "|---|" + "---|" * 4]
    for m in METHODS[1:]:
        L.append(f"| {m} − none | " + " | ".join("N/A" if M["delta_pp_clean"][m][k] is None else f"{M['delta_pp_clean'][m][k]:+.2f}" for k in KEYS) + " |")
    L += ["", "## Mixed-noise input, per seed", "", "| Method | Seed | " + " | ".join(NAMES[k] for k in KEYS) + " |", "|---|---|" + "---|" * 4]
    for m in METHODS:
        for s in SEEDS:
            L.append(f"| {m} | {s} | " + " | ".join(fmt(M["noisy_seeds"][m][s][k]) for k in KEYS) + " |")
    L += ["", "## Mixed-noise input, mean ± std over seeds 42/43/44 (sample std)", "",
          "| Method | " + " | ".join(NAMES[k] for k in KEYS) + " |", "|---|" + "---|" * 4]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(ms(*M["noisy_mean_std"][m][k]) for k in KEYS) + " |")
    L += ["", "Δ vs no-filter, paired by seed, mean ± std (pp):", "",
          "| Method | " + " | ".join(NAMES[k] for k in KEYS) + " |", "|---|" + "---|" * 4]
    for m in METHODS[1:]:
        cells = []
        for k in KEYS:
            mu_, sd = M["delta_pp_noisy_paired"][m][k]
            cells.append("N/A" if mu_ is None else f"{mu_:+.2f} ± {sd:.2f}")
        L.append(f"| {m} − none | " + " | ".join(cells) + " |")
    L += ["", "## Timing (ms per image, averaged over all passes of the method)", "",
          "| Method | filter | inference |", "|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | {T[m]['filter_ms_per_img']:.1f} | {T[m]['infer_ms_per_img']:.1f} |")
    L.append(f"\nNoise generation (not part of filter time): {T['noise_gen_ms_per_noisy_image']:.1f} ms per noisy image.")
    L += ["", "## Per-class mask AP (%)", "",
          "N/A = no GT instance of the class in the evaluated images. Noisy columns = mean over 3 seeds (std in `per_class.csv`).", "",
          "| Class | GT n | Clean: none | median | bilateral | ARC | Noisy: none | median | bilateral | ARC |", "|---|---|" + "---|" * 8]
    for c in sorted(cats):
        cl = [fmt(pc[f"clean_{m}"][c]) for m in METHODS]
        nz = [ms(mean_std([pc[p][c] for p in seedp(m)])[0]) for m in METHODS]
        L.append(f"| {cats[c]} | {gt_n[c]} | " + " | ".join(cl + nz) + " |")
    (out / "results_table.md").write_text("\n".join(L) + "\n")
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2))
    log("report written")
    print("\n".join(L[:60]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "arc_full_valid"),
                    help="output directory (default: results/arc_full_valid)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N validation images (smoke test)")
    ap.add_argument("--stage", default="all", choices=["all", "run", "eval", "report"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2))
    if a.stage in ("all", "run"): run_stage(out, a.limit)
    if a.stage in ("all", "eval"): eval_stage(out, a.limit, a.workers)
    if a.stage in ("all", "report"): report_stage(out)


if __name__ == "__main__":
    main()

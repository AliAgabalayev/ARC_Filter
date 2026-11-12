"""Experimental ARC13 Fashionpedia pilot (30 images; clean and seed-42 noise).

Requires data/fashionpedia_2020_v1_split/valid/_annotations.coco.json and images,
plus checkpoints/rfdetr_fashionpedia_b4.pth for inference at resolution 768.
Use --image-ids /external/ids.json for a JSON list of exactly 30 unique integer
validation IDs; the legacy results/arc_pilot/image_ids.json remains the default.
All generated artifacts belong in an external --out directory.

Pilot: python scripts/arc13_pilot.py --image-ids /external/ids.json --out /external/run
Replay: python scripts/arc13_pilot.py --evaluate-only --out /external/run
        --replay-out /external/replay
Replay uses saved predictions, config.json, image_ids.json and hash-verified
source/ snapshots plus matching local annotations; no checkpoint or model is
loaded. Replay outputs must be fresh. --image-ids applies only to new inference.
"""
import argparse
import ast
import importlib.metadata
from datetime import datetime, timezone
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc13_filter import ARC13_filter
from arc_common import (ROOT, CHECKPOINT, VALID_ANN, VALID_DIR, add_mixed_noise,
                        check_class_mapping, evaluate, load_model, predict, to_rle, to_uint8)
from arc_full_valid import filter_rgb as baseline_filter

METHODS = ['none', 'median', 'arc9', 'arc13_trim2', 'arc13_trim3']
CONDITIONS = ['clean', 'noisy42']
CUTOFFS = [.001, .01, .05]
KEYS = ['AP', 'AP50', 'AP75', 'AP_small']


def filter_rgb(method, rgb):
    if method in ('none', 'median', 'arc9'):
        return baseline_filter('arc' if method == 'arc9' else method, rgb)
    hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    v = ARC13_filter(hsv[:, :, 2].astype(np.float64), trim=int(method[-1]))
    hsv[:, :, 2] = np.clip(v, 0, 1).astype(np.float32)
    return to_uint8(np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def replay(experiment, destination):
    """Validate archived inference provenance; write evaluation provenance separately."""
    config = json.loads((experiment / 'config.json').read_text())
    ids = json.loads((experiment / 'image_ids.json').read_text())
    expected = dict(methods=METHODS, conditions=CONDITIONS, cutoffs=CUTOFFS,
                    resolution=768, seed=42, prediction_threshold=.001,
                    strict_cutoffs=True, maxDets=[1, 10, 100], num_select=100)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Incompatible inference configuration')
    if ids != config['image_ids'] or len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError('Incompatible saved image IDs')
    if digest(VALID_ANN) != config['annotation_sha256']:
        raise ValueError('Annotation hash mismatch')
    for name, sha in config['source_sha256'].items():
        archived = experiment / 'source' / name
        if digest(archived) != sha:
            raise ValueError(f'Archived inference source hash mismatch: {name}')
        if name != 'scripts/arc13_pilot.py' and digest(ROOT / name) != sha:
            raise ValueError(f'Inference dependency changed: {name}')
    # Compare the actual prediction-producing branch and filter, not the runner file hash.
    def inference_parts(path):
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        branch = next(n for n in main.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'args.evaluate_only' and n.orelse)
        filt = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'filter_rgb')
        configuration = next(n for n in main.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
        return [ast.dump(n, include_attributes=False) for n in [filt, configuration, *branch.orelse]]
    if inference_parts(experiment/'source/scripts/arc13_pilot.py') != inference_parts(Path(__file__)):
        raise ValueError('Inference-affecting runner change')
    archived_tree = ast.parse((experiment/'source/scripts/arc13_pilot.py').read_text())
    archived_main = next(n for n in archived_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    expression = next(n.value for n in archived_main.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
    def recorded_digest(path):
        if Path(path) == CHECKPOINT:
            return config['checkpoint_sha256']  # No checkpoint access during replay.
        if Path(path) == VALID_ANN:
            return config['annotation_sha256']
        return config['source_sha256'][str(Path(path).relative_to(ROOT))]
    namespace = dict(globals(), ids=ids, source=Path(config['image_ids_source']),
                     digest=recorded_digest)
    # Execute only the config expression from the hash-verified local source snapshot.
    original_config = eval(compile(ast.Expression(expression), '<archived-config>', 'eval'), namespace)
    if any(config.get(k) != v for k, v in original_config.items()):
        raise ValueError('Incompatible archived inference configuration')
    coco = COCO(str(VALID_ANN))
    if config['categories'] != coco.dataset['categories'] or not set(ids) <= set(coco.getImgIds()):
        raise ValueError('Category mapping or image IDs mismatch')
    with gzip.open(experiment/'predictions.json.gz', 'rt') as f:
        dets = json.load(f)
    if set(dets) != {f'{c}_{m}' for c in CONDITIONS for m in METHODS}:
        raise ValueError('Incompatible prediction sets')
    for rows in dets.values():
        for d in rows:
            rec = coco.imgs.get(d['image_id'])
            if (d['image_id'] not in ids or d['category_id'] not in coco.cats
                    or not np.isfinite(d['score']) or not .001 < d['score'] <= 1
                    or d['segmentation']['size'] != [rec['height'], rec['width']]):
                raise ValueError('Invalid cached prediction')
    destination = (destination or experiment / ('evaluation_replay_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if destination == ROOT or ROOT in destination.parents or destination == experiment:
        raise ValueError('Replay output must be a fresh external directory')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(command=[sys.executable, *sys.argv], utc=datetime.now(timezone.utc).isoformat(),
                    inference_config_sha256=digest(experiment/'config.json'),
                    prediction_sha256=digest(experiment/'predictions.json.gz'),
                    annotation_sha256=digest(VALID_ANN), image_ids=ids,
                    archived_sources=config['source_sha256'],
                    evaluator_sources={str(p): digest(p) for p in [Path(__file__).resolve(), ROOT/'scripts/arc_common.py']},
                    versions={k: importlib.metadata.version(k) for k in ['numpy', 'pycocotools', 'scipy']},
                    inference_branch_matches_archive=True, model_loaded=False, inference_ran=False)
    save(destination/'manifest.json', manifest)
    evaluate_cache(coco, dets, ids, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='External run directory (existing experiment for replay)')
    parser.add_argument('--image-ids', type=Path, help='External JSON list of exactly 30 validation image IDs')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--replay-out', type=Path, help='Fresh external evaluation output directory')
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Output directory must be outside repository')
    if args.evaluate_only:
        if args.image_ids is not None:
            parser.error('--image-ids applies only to new inference; replay uses saved IDs')
        replay(out, args.replay_out)
        return
    if args.replay_out is not None:
        parser.error('--replay-out requires --evaluate-only')
    import torch
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    source = args.image_ids.resolve() if args.image_ids else ROOT / 'results/arc_pilot/image_ids.json'
    ids = json.loads(source.read_text())
    if (not isinstance(ids, list) or len(ids) != 30
            or any(type(i) is not int for i in ids) or len(set(ids)) != 30):
        raise ValueError('Expected exactly 30 unique integer validation image IDs')
    coco = COCO(str(VALID_ANN))
    recs = {r['id']: r for r in coco.dataset['images']}
    if not set(ids) <= set(recs):
        raise ValueError('Image IDs must belong to the validation annotations')
    config = dict(image_ids=ids, image_ids_source=str(source), resolution=768,
                  checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
                  annotation_sha256=digest(VALID_ANN), methods=METHODS,
                  conditions=CONDITIONS, cutoffs=CUTOFFS, prediction_threshold=.001,
                  filters={'median': 'OpenCV 3x3 V-channel replicate padding', 'arc9': 'Original 3x3, trim1, median9, Otsu7, symmetric padding', 'arc13_trim2': '3x3 plus axial distance2, median13, trim2, Otsu9, symmetric padding', 'arc13_trim3': 'same13, median13, trim3, Otsu7, symmetric padding'},
                  torch_threads=4, topk='Global query-class top-k before strict score cutoff; capped candidate pool', strict_cutoffs=True, maxDets=[1, 10, 100], seed=42,
                  noise='default_rng([42,image_id]); Gaussian sigma .10, clip; 1% spatial salt/pepper equally likely across all channels',
                  timing='perf_counter; CUDA synchronization around predict if available; filter includes HSV/RGB and uint8; inference includes preprocessing and mask upsampling; excludes RLE encoding; one warmup excluded',
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [ROOT/'arc_filter.py', ROOT/'arc13_filter.py', Path(__file__).resolve(), ROOT/'scripts/arc_common.py', ROOT/'scripts/arc_full_valid.py']})
    config_path = out / 'config.json'
    cache = out / 'predictions.json.gz'
    if not config_path.exists() and not cache.exists():
        for name in config['source_sha256']:
            snapshot = out / 'source' / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / name).read_bytes())
    if args.evaluate_only:
        previous = json.loads(config_path.read_text())
        assert all(previous[k] == v for k, v in config.items()), 'Configuration mismatch'
        with gzip.open(cache, 'rt') as f:
            dets = json.load(f)
    else:
        if cache.exists() or config_path.exists():
            raise FileExistsError('Use a fresh output directory; caches are never silently reused')
        save(out/'image_ids.json', ids)
        save(config_path, config)
        model = load_model()
        check_class_mapping(model, coco)
        config.update(num_select=int(model.model.postprocess.num_select), device=str(model.model.device),
                      torch_version=torch.__version__, numpy_version=np.__version__, categories=coco.dataset['categories'])
        save(config_path, config)
        print('Model configuration:', config['device'], 'num_select=', config['num_select'], flush=True)
        sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        first = np.asarray(Image.open(VALID_DIR/recs[ids[0]]['file_name']).convert('RGB'))
        predict(model, first, .001)
        sync()
        dets = {f'{c}_{m}': [] for c in CONDITIONS for m in METHODS}
        timings = []
        for n, iid in enumerate(ids):
            rec = recs[iid]
            rgb = np.asarray(Image.open(VALID_DIR/rec['file_name']).convert('RGB'), np.float32)/255
            assert rgb.shape[:2] == (rec['height'], rec['width'])
            noisy = add_mixed_noise(rgb, iid, 42)
            for c, arr in [('clean', rgb), ('noisy42', noisy)]:
                for m in METHODS:
                    start = time.perf_counter()
                    inp = filter_rgb(m, arr)
                    filter_s = time.perf_counter()-start
                    sync()
                    start = time.perf_counter()
                    pr = predict(model, inp, .001)
                    sync()
                    infer_s = time.perf_counter()-start
                    assert len(pr['scores']) <= config['num_select']
                    for mask, cid, score in zip(pr['masks'], pr['class_ids'], pr['scores']):
                        rle = to_rle(mask)
                        assert rle['size'] == [rec['height'], rec['width']]
                        assert np.array_equal(mask_util.decode(rle).astype(bool), mask)
                        dets[f'{c}_{m}'].append(dict(image_id=iid, category_id=int(cid)+1, score=float(score), segmentation=rle))
                    timings.append(dict(image_id=iid, condition=c, method=m, filter_s=filter_s, inference_s=infer_s))
                    print(n+1, iid, c, m, f'filter={filter_s:.3f}s inference={infer_s:.3f}s', flush=True)
            save(out/'timings.json', timings)
            with gzip.open(out/f'partial_{iid}.json.gz', 'wt') as f:
                json.dump({k:[d for d in v if d['image_id']==iid] for k,v in dets.items()}, f)
        with gzip.open(cache, 'wt') as f:
            json.dump(dets, f)
        for iid in ids:
            (out/f'partial_{iid}.json.gz').unlink()
    evaluate_cache(coco, dets, ids, out)


def evaluate_cache(coco, dets, ids, out):
    metrics = {}
    for key, predictions in dets.items():
        for cutoff in CUTOFFS:
            selected = [dict(d) for d in predictions if d['score'] > cutoff]
            metrics[f'{key}|{cutoff}'] = evaluate(coco, selected, ids, out/f'cocoeval_{key}_{cutoff}.log')
            metrics[f'{key}|{cutoff}']['n_predictions'] = len(selected)
            result = metrics[f'{key}|{cutoff}']
            result['per_class'] = {int(k): v for k, v in result['per_class'].items()}
    deltas = {}
    for c in CONDITIONS:
        for m in METHODS:
            for cutoff in CUTOFFS:
                a = metrics[f'{c}_{m}|{cutoff}']
                for base in ['arc9', 'none']:
                    b = metrics[f'{c}_{base}|{cutoff}']
                    deltas[f'{c}_{m}-vs-{base}|{cutoff}'] = {k: None if a[k] is None or b[k] is None else 100*(a[k]-b[k]) for k in KEYS}
    counts = {cat['id']: dict(name=cat['name'], gt_instances=len(coco.getAnnIds(imgIds=ids, catIds=[cat['id']]))) for cat in coco.dataset['categories']}
    save(out/'metrics.json', dict(metrics=metrics, differences_pp=deltas, categories=counts))
    print('Completed ten passes and thirty evaluations:', out, flush=True)


if __name__ == '__main__':
    main()

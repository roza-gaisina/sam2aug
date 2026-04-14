#!/usr/bin/env python3
"""
Unified evaluator for the OBJECT RESCALE experiment.

Supported modes
---------------
- original  : evaluate the untouched original curated ImageNet subset
- generated : evaluate generated same-background relocated images across scales
- compare   : merge original + generated results and compute paired deltas

Expected generation layout
--------------------------
<paths.outputs.root>/<experiment.name>/
  images/
    scale100/<wnid>/<image_id>__scale-100.jpg
    scale075/<wnid>/<image_id>__scale-075.jpg
    scale050/<wnid>/<image_id>__scale-050.jpg
    scale025/<wnid>/<image_id>__scale-025.jpg
  meta/
    same_background_scale_gen_manifest.json

Expected config pattern
-----------------------
- supports YAML inheritance via `inherits`
- reuses:
    paths.imagenet.val_root
    paths.outputs.root
    content.manifest
    experiment.name
    evaluation.* (optional)

Main metrics
------------
Per-image:
- correct_in_top1
- correct_in_top5
- correct_rank
- prob_correct
- logit_correct
- prob_top1
- top1_logit_minus_correct_logit
- pred_changed_vs_original (compare mode)
- delta_prob_correct_vs_original (compare mode)
- delta_logit_correct_vs_original (compare mode)
- delta_rank_vs_original (compare mode)

Compare outputs:
- paired CSV
- summary by scale
- summary by class x scale

Example usage
-------------
python evaluate_inpainting_rescale_size_for_prediction.py --config configs/inpainting_rescale_size_for_prediction.yaml --mode original --model vit_base_patch16_224

python evaluate_inpainting_rescale_size_for_prediction.py --config configs/inpainting_rescale_size_for_prediction.yaml --mode generated --model vit_base_patch16_224

python evaluate_inpainting_rescale_size_for_prediction.py --config configs/inpainting_rescale_size_for_prediction.yaml --mode compare --model vit_base_patch16_224

Useful timm model names
-----------------------
- resnet18.a1_in1k
- resnet50.a1_in1k
- vit_tiny_patch16_224.augreg_in21k_ft_in1k
- vit_base_patch16_224.augreg_in21k_ft_in1k
  
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

import torch
import torch.nn.functional as F

import timm
from timm.data import create_transform, resolve_data_config

try:
    import yaml
except ImportError as e:
    raise ImportError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from e


DEFAULT_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"


# ----------------------------
# Config helpers
# ----------------------------
def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping/dict at top level.")
    return data


def load_config_with_inherits(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = cfg.get("inherits", None)
    if not parent:
        return cfg

    if not isinstance(parent, str):
        raise ValueError("'inherits' must be a string path to a YAML file.")

    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(config_path), parent)
    base_cfg = load_yaml(parent_path)

    child_cfg = dict(cfg)
    child_cfg.pop("inherits", None)

    merged = deep_update(base_cfg, child_cfg)
    merged["_meta"] = {
        "config_path": config_path,
        "base_path": parent_path,
    }
    return merged


# ----------------------------
# ImageNet class index mapping
# ----------------------------
def ensure_imagenet_class_index(path: str, url: str = DEFAULT_CLASS_INDEX_URL) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    print(f"[INFO] Downloading imagenet_class_index.json -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def load_class_index_json(path: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str]]:
    with open(path, "r") as f:
        data = json.load(f)

    idx_to_wnid: Dict[int, str] = {}
    idx_to_label: Dict[int, str] = {}
    for k, (wnid, label) in data.items():
        i = int(k)
        idx_to_wnid[i] = wnid
        idx_to_label[i] = label

    wnid_to_idx = {wnid: i for i, wnid in idx_to_wnid.items()}
    return idx_to_wnid, wnid_to_idx, idx_to_label


# ----------------------------
# Data model
# ----------------------------
def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class Sample:
    image_id: str
    class_name: str
    wnid: str
    img_path: str
    condition_key: Tuple[str, str]  # (class_name, condition)
    scale_factor: Optional[float] = None
    scale_pct: Optional[int] = None
    scale_tag: Optional[str] = None
    original_image_id: Optional[str] = None


def iter_batches(samples: List[Sample], transform, batch_size: int):
    batch = []
    for s in samples:
        if not os.path.exists(s.img_path):
            raise FileNotFoundError(f"Missing image: {s.img_path}")
        img = load_rgb(s.img_path)
        x = transform(img)
        batch.append((s, x))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------------------
# Sample builders
# ----------------------------
def build_samples_original(content_manifest_path: str, val_root: str) -> List[Sample]:
    with open(content_manifest_path, "r") as f:
        m = json.load(f)

    samples: List[Sample] = []
    for e in m:
        image_id = e.get("image_id") or os.path.splitext(os.path.basename(e["original_rel_path"]))[0]
        class_name = e["class_name"]
        wnid = e["wnid"]
        img_path = os.path.join(val_root, e["original_rel_path"])

        samples.append(
            Sample(
                image_id=image_id,
                class_name=class_name,
                wnid=wnid,
                img_path=img_path,
                condition_key=(class_name, "original"),
            )
        )
    return samples


def build_samples_generated(gen_manifest_path: str, out_root: str) -> List[Sample]:
    with open(gen_manifest_path, "r") as f:
        m = json.load(f)

    samples: List[Sample] = []
    for e in m:
        image_id = (
            e.get("image_id")
            or e.get("original_image_id")
            or os.path.splitext(os.path.basename(e["output_rel_path"]))[0]
        )

        # support both schemas:
        # - class_name / wnid
        # - shape_name / shape_wnid
        class_name = e.get("class_name", e.get("shape_name"))
        wnid = e.get("wnid", e.get("shape_wnid"))

        if class_name is None:
            raise KeyError(
                "Generated manifest entry is missing both 'class_name' and 'shape_name'."
            )
        if wnid is None:
            raise KeyError(
                "Generated manifest entry is missing both 'wnid' and 'shape_wnid'."
            )

        scale_factor = e.get("scale_factor")
        scale_pct = e.get("scale_pct")
        scale_tag = e.get("scale_tag")
        original_image_id = e.get("original_image_id") or e.get("image_id")
        output_rel_path = e["output_rel_path"]
        img_path = os.path.join(out_root, output_rel_path)

        # robust fallback if manifest does not have explicit scale_tag
        if scale_tag is None:
            if scale_pct is not None:
                scale_tag = f"scale{int(scale_pct):03d}"
            elif scale_factor is not None:
                scale_tag = f"scale{int(round(float(scale_factor) * 100)):03d}"
            else:
                p = Path(output_rel_path)
                parts = p.parts
                scale_tag = parts[1] if len(parts) >= 2 else "generated"

        samples.append(
            Sample(
                image_id=f"{original_image_id}__{scale_tag}",
                class_name=class_name,
                wnid=wnid,
                img_path=img_path,
                condition_key=(class_name, scale_tag),
                scale_factor=float(scale_factor) if scale_factor is not None else None,
                scale_pct=int(scale_pct) if scale_pct is not None else None,
                scale_tag=str(scale_tag),
                original_image_id=original_image_id,
            )
        )
    return samples

# ----------------------------
# Core evaluation
# ----------------------------
def evaluate_samples(
    samples: List[Sample],
    model,
    transform,
    device,
    batch_size: int,
    topk: int,
    idx_to_wnid: Dict[int, str],
    wnid_to_idx: Dict[str, int],
    idx_to_label: Dict[int, str],
    mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in iter_batches(samples, transform, batch_size):
            batch_samples, batch_tensors = zip(*batch)
            x = torch.stack(batch_tensors, dim=0).to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            topk_prob, topk_idx = probs.topk(topk, dim=1)
            top1_idx = topk_idx[:, 0]
            top1_prob = topk_prob[:, 0]
            top1_logit = logits.gather(1, top1_idx.view(-1, 1)).squeeze(1)

            for i, s in enumerate(batch_samples):
                target_idx = wnid_to_idx[s.wnid]
                pred_i = int(top1_idx[i].item())
                pred_wnid = idx_to_wnid[pred_i]
                pred_label = idx_to_label[pred_i]

                topk_list = [int(v) for v in topk_idx[i].detach().cpu().tolist()]
                correct_in_top1 = int(pred_i == target_idx)
                correct_in_top5 = int(target_idx in topk_list)
                correct_rank = topk_list.index(target_idx) + 1 if target_idx in topk_list else (topk + 1)

                logit_correct = float(logits[i, target_idx].item())
                prob_correct = float(probs[i, target_idx].item())
                prob_top1 = float(top1_prob[i].item())
                top1_minus_correct = float(top1_logit[i].item() - logit_correct)

                row = {
                    "mode": mode,
                    "image_id": s.image_id,
                    "original_image_id": s.original_image_id or s.image_id,
                    "class_name": s.class_name,
                    "wnid": s.wnid,
                    "scale_factor": s.scale_factor,
                    "scale_pct": s.scale_pct,
                    "scale_tag": s.scale_tag,
                    "pred_idx": pred_i,
                    "pred_wnid": pred_wnid,
                    "pred_label": pred_label,
                    "target_idx": target_idx,
                    "correct_in_top1": correct_in_top1,
                    "correct_in_top5": correct_in_top5,
                    "correct_rank": correct_rank,
                    "logit_correct": logit_correct,
                    "prob_correct": prob_correct,
                    "prob_top1": prob_top1,
                    "top1_logit_minus_correct_logit": top1_minus_correct,
                    "img_path": s.img_path,
                }
                rows.append(row)

    return rows


# ----------------------------
# CSV / summary helpers
# ----------------------------
def write_csv(path: str, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for CSV: {path}")
    if header is None:
        header = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_scale(compare_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in compare_rows:
        groups[str(r["scale_tag"])].append(r)

    out = []
    for scale_tag, rows in sorted(groups.items()):
        n = len(rows)
        out.append({
            "scale_tag": scale_tag,
            "scale_factor": rows[0].get("scale_factor"),
            "scale_pct": rows[0].get("scale_pct"),
            "n": n,
            "top1_acc_original": sum(float(r["correct_in_top1_original"]) for r in rows) / n,
            "top1_acc_generated": sum(float(r["correct_in_top1_generated"]) for r in rows) / n,
            "top5_acc_original": sum(float(r["correct_in_top5_original"]) for r in rows) / n,
            "top5_acc_generated": sum(float(r["correct_in_top5_generated"]) for r in rows) / n,
            "mean_prob_correct_original": sum(float(r["prob_correct_original"]) for r in rows) / n,
            "mean_prob_correct_generated": sum(float(r["prob_correct_generated"]) for r in rows) / n,
            "mean_delta_prob_correct": sum(float(r["delta_prob_correct_vs_original"]) for r in rows) / n,
            "mean_delta_logit_correct": sum(float(r["delta_logit_correct_vs_original"]) for r in rows) / n,
            "mean_delta_rank": sum(float(r["delta_rank_vs_original"]) for r in rows) / n,
            "pred_changed_rate": sum(float(r["pred_changed_vs_original"]) for r in rows) / n,
        })
    return out


def summarize_by_class_and_scale(compare_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in compare_rows:
        groups[(str(r["class_name"]), str(r["scale_tag"]))].append(r)

    out = []
    for (class_name, scale_tag), rows in sorted(groups.items()):
        n = len(rows)
        out.append({
            "class_name": class_name,
            "scale_tag": scale_tag,
            "scale_factor": rows[0].get("scale_factor"),
            "scale_pct": rows[0].get("scale_pct"),
            "n": n,
            "top1_acc_original": sum(float(r["correct_in_top1_original"]) for r in rows) / n,
            "top1_acc_generated": sum(float(r["correct_in_top1_generated"]) for r in rows) / n,
            "top5_acc_original": sum(float(r["correct_in_top5_original"]) for r in rows) / n,
            "top5_acc_generated": sum(float(r["correct_in_top5_generated"]) for r in rows) / n,
            "mean_prob_correct_original": sum(float(r["prob_correct_original"]) for r in rows) / n,
            "mean_prob_correct_generated": sum(float(r["prob_correct_generated"]) for r in rows) / n,
            "mean_delta_prob_correct": sum(float(r["delta_prob_correct_vs_original"]) for r in rows) / n,
            "pred_changed_rate": sum(float(r["pred_changed_vs_original"]) for r in rows) / n,
        })
    return out


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def coerce_optional_float(v: Any) -> Optional[float]:
    if v in (None, "", "None"):
        return None
    return float(v)


def coerce_optional_int(v: Any) -> Optional[int]:
    if v in (None, "", "None"):
        return None
    return int(float(v))


def merge_original_and_generated(original_rows: List[Dict[str, Any]], generated_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    orig_map = {str(r["image_id"]): r for r in original_rows}
    merged: List[Dict[str, Any]] = []

    for g in generated_rows:
        orig_id = str(g["original_image_id"])
        if orig_id not in orig_map:
            raise KeyError(f"Generated row references original_image_id={orig_id}, but no original row found.")
        o = orig_map[orig_id]

        row = {
            "original_image_id": orig_id,
            "generated_image_id": g["image_id"],
            "class_name": g["class_name"],
            "wnid": g["wnid"],
            "scale_tag": g.get("scale_tag"),
            "scale_factor": coerce_optional_float(g.get("scale_factor")),
            "scale_pct": coerce_optional_int(g.get("scale_pct")),
            "pred_wnid_original": o["pred_wnid"],
            "pred_label_original": o["pred_label"],
            "pred_wnid_generated": g["pred_wnid"],
            "pred_label_generated": g["pred_label"],
            "correct_in_top1_original": int(o["correct_in_top1"]),
            "correct_in_top1_generated": int(g["correct_in_top1"]),
            "correct_in_top5_original": int(o["correct_in_top5"]),
            "correct_in_top5_generated": int(g["correct_in_top5"]),
            "correct_rank_original": int(o["correct_rank"]),
            "correct_rank_generated": int(g["correct_rank"]),
            "prob_correct_original": float(o["prob_correct"]),
            "prob_correct_generated": float(g["prob_correct"]),
            "logit_correct_original": float(o["logit_correct"]),
            "logit_correct_generated": float(g["logit_correct"]),
            "prob_top1_original": float(o["prob_top1"]),
            "prob_top1_generated": float(g["prob_top1"]),
            "pred_changed_vs_original": int(o["pred_wnid"] != g["pred_wnid"]),
            "delta_prob_correct_vs_original": float(g["prob_correct"]) - float(o["prob_correct"]),
            "delta_logit_correct_vs_original": float(g["logit_correct"]) - float(o["logit_correct"]),
            "delta_rank_vs_original": int(g["correct_rank"]) - int(o["correct_rank"]),
            "img_path_original": o["img_path"],
            "img_path_generated": g["img_path"],
        }
        merged.append(row)

    return merged


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Experiment YAML (e.g. geirhos16_same_background_scale_center.yaml).")
    ap.add_argument("--mode", required=True, choices=["original", "generated", "compare"],
                    help="Evaluate original images, generated images, or compare paired CSVs.")
    ap.add_argument("--model", default=None, help="Override evaluation.model (e.g. resnet18.a1_in1k).")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Override evaluation.device.")
    ap.add_argument("--batch-size", type=int, default=None, help="Override evaluation.batch_size.")
    ap.add_argument("--topk", type=int, default=None, help="Override evaluation.topk.")
    ap.add_argument("--csv-out", default=None, help="Override output CSV path.")
    ap.add_argument("--class-index-json", default=None, help="Override imagenet_class_index.json path.")
    ap.add_argument("--gen-manifest", default=None,
                    help="(generated mode) override gen manifest path. Default: <out_root>/meta/rescale_size_for_prediction_gen_manifest.json")
    ap.add_argument("--original-csv", default=None,
                    help="(compare mode) override original CSV path.")
    ap.add_argument("--generated-csv", default=None,
                    help="(compare mode) override generated CSV path.")
    args = ap.parse_args()

    cfg = load_config_with_inherits(args.config)

    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    out_root = os.path.join(cfg["paths"]["outputs"]["root"], exp_name)
    meta_dir = os.path.join(out_root, "meta")
    Path(meta_dir).mkdir(parents=True, exist_ok=True)

    val_root = cfg["paths"]["imagenet"]["val_root"]
    content_manifest = cfg["content"]["manifest"]

    ev = cfg.get("evaluation", {})
    model_name = args.model or ev.get("model", "resnet18.a1_in1k")
    device_req = args.device or ev.get("device", "cuda")
    batch_size = int(args.batch_size or ev.get("batch_size", 32))
    topk = int(args.topk or ev.get("topk", 5))

    safe_model_tag = model_name.replace('.', '_').replace('/', '_')
    class_index_json = args.class_index_json or ev.get("class_index_json") or os.path.join(meta_dir, "imagenet_class_index.json")

    default_gen_manifest = os.path.join(meta_dir, "rescale_size_for_prediction_gen_manifest.json")
    gen_manifest = args.gen_manifest or ev.get("gen_manifest") or default_gen_manifest

    default_original_csv = os.path.join(meta_dir, f"eval_original_{safe_model_tag}_top{topk}.csv")
    default_generated_csv = os.path.join(meta_dir, f"eval_rescale_size_for_prediction_{safe_model_tag}_top{topk}.csv")
    default_compare_csv = os.path.join(meta_dir, f"eval_rescale_size_for_prediction_compare_{safe_model_tag}_top{topk}.csv")
    default_summary_scale_csv = os.path.join(meta_dir, f"eval_rescale_size_for_prediction_summary_by_scale_{safe_model_tag}_top{topk}.csv")
    default_summary_class_scale_csv = os.path.join(meta_dir, f"eval_rescale_size_for_prediction_summary_by_class_scale_{safe_model_tag}_top{topk}.csv")

    csv_out = args.csv_out or ev.get("csv_out")

    if args.mode == "compare":
        original_csv = args.original_csv or default_original_csv
        generated_csv = args.generated_csv or default_generated_csv
        compare_csv = csv_out or default_compare_csv
        summary_scale_csv = os.path.join(meta_dir, Path(default_summary_scale_csv).name)
        summary_class_scale_csv = os.path.join(meta_dir, Path(default_summary_class_scale_csv).name)

        print("[INFO] Mode: compare")
        print("[INFO] original_csv:", original_csv)
        print("[INFO] generated_csv:", generated_csv)
        print("[INFO] compare_csv:", compare_csv)

        if not os.path.exists(original_csv):
            raise FileNotFoundError(f"Original CSV not found: {original_csv}")
        if not os.path.exists(generated_csv):
            raise FileNotFoundError(f"Generated CSV not found: {generated_csv}")

        original_rows = read_csv_rows(original_csv)
        generated_rows = read_csv_rows(generated_csv)
        compare_rows = merge_original_and_generated(original_rows, generated_rows)
        summary_scale_rows = summarize_by_scale(compare_rows)
        summary_class_scale_rows = summarize_by_class_and_scale(compare_rows)

        write_csv(compare_csv, compare_rows)
        write_csv(summary_scale_csv, summary_scale_rows)
        write_csv(summary_class_scale_csv, summary_class_scale_rows)

        print(f"[INFO] Wrote compare CSV: {compare_csv}")
        print(f"[INFO] Wrote summary by scale CSV: {summary_scale_csv}")
        print(f"[INFO] Wrote summary by class+scale CSV: {summary_class_scale_csv}")
        return

    device = torch.device("cuda" if (device_req == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Mode:", args.mode)
    print("[INFO] Experiment out_root:", out_root)
    print("[INFO] Model:", model_name)
    print("[INFO] Batch_size:", batch_size, "topk:", topk)
    print("[INFO] csv_out:", csv_out)

    ensure_imagenet_class_index(class_index_json)
    idx_to_wnid, wnid_to_idx, idx_to_label = load_class_index_json(class_index_json)

    if args.mode == "original":
        samples = build_samples_original(content_manifest, val_root)
        default_csv = default_original_csv
    else:
        if not os.path.exists(gen_manifest):
            raise FileNotFoundError(f"Gen manifest not found: {gen_manifest}")
        samples = build_samples_generated(gen_manifest, out_root)
        default_csv = default_generated_csv
        print("[INFO] gen_manifest:", gen_manifest)

    csv_out = csv_out or default_csv

    print(f"[INFO] Loaded {len(samples)} samples")

    missing_wnids = sorted({s.wnid for s in samples if s.wnid not in wnid_to_idx})
    if missing_wnids:
        raise KeyError(f"Some WNIDs are missing from imagenet_class_index.json: {missing_wnids[:25]}")

    print(f"[INFO] Loading timm model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)

    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg, is_training=False)

    rows = evaluate_samples(
        samples=samples,
        model=model,
        transform=transform,
        device=device,
        batch_size=batch_size,
        topk=topk,
        idx_to_wnid=idx_to_wnid,
        wnid_to_idx=wnid_to_idx,
        idx_to_label=idx_to_label,
        mode=args.mode,
    )

    write_csv(csv_out, rows)
    print(f"[INFO] Wrote {len(rows)} rows -> {csv_out}")

    # Small console summary
    if args.mode == "original":
        top1 = sum(int(r["correct_in_top1"]) for r in rows) / len(rows)
        top5 = sum(int(r["correct_in_top5"]) for r in rows) / len(rows)
        print(f"[INFO] Original top-1: {top1:.4f} | top-{topk}: {top5:.4f}")
    else:
        by_scale_top1: Dict[str, List[int]] = defaultdict(list)
        by_scale_top5: Dict[str, List[int]] = defaultdict(list)
        for r in rows:
            by_scale_top1[str(r["scale_tag"])].append(int(r["correct_in_top1"]))
            by_scale_top5[str(r["scale_tag"])].append(int(r["correct_in_top5"]))
        for scale_tag in sorted(by_scale_top1.keys()):
            t1 = sum(by_scale_top1[scale_tag]) / len(by_scale_top1[scale_tag])
            t5 = sum(by_scale_top5[scale_tag]) / len(by_scale_top5[scale_tag])
            print(f"[INFO] {scale_tag}: top-1={t1:.4f} | top-{topk}={t5:.4f} | n={len(by_scale_top1[scale_tag])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified evaluator for Geirhos16 background-shift experiments.

Supported modes
---------------
1) --mode original
   Evaluate the original curated Geirhos16 subset from content.manifest.

2) --mode background_shift
   Evaluate generated background-shift images from a generation manifest.

3) --mode compare
   Merge an original CSV and a generated/background-shift CSV and compute
   paired deltas against the original source image.

The script is adapted from the texture-only evaluator, but uses metrics that
fit the background-shift setting better:
- correct class top-1 / top-k
- correct class rank
- correct class probability / logit
- confidence drop relative to originals (compare mode)

Typical usage
-------------
# Evaluate originals
python evaluate_object_relocation_background_shift.py --config configs/object_relocation_background_shift.yaml --mode original --model vit_base_patch16_224

# Evaluate generated background-shift images
python evaluate_object_relocation_background_shift.py --config configs/object_relocation_background_shift.yaml --mode background_shift --model vit_base_patch16_224

# Compare generated vs original
python evaluate_object_relocation_background_shift.py --config configs/object_relocation_background_shift.yaml --mode compare --model vit_base_patch16_224

Useful timm model names
-----------------------
- resnet18.a1_in1k
- resnet50.a1_in1k
- vit_tiny_patch16_224.augreg_in21k_ft_in1k
- vit_base_patch16_224.augreg_in21k_ft_in1k
- vit_tiny_patch16_224
- vit_base_patch16_224
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
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
DEFAULT_GEN_MANIFEST_CANDIDATES = [
    "background_shift_manifest.json"
]


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# ImageNet class index mapping
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class Sample:
    sample_id: str
    source_image_id: str
    class_name: str
    wnid: str
    img_path: str
    mode: str
    background_name: str = "original"
    output_rel_path: str = ""
    content_rel_path: str = ""
    condition_key: Tuple[str, str] = ("", "")  # (class_name, background_name)


# -----------------------------------------------------------------------------
# Sample builders
# -----------------------------------------------------------------------------
def build_samples_original(content_manifest_path: str, val_root: str) -> List[Sample]:
    with open(content_manifest_path, "r") as f:
        m = json.load(f)

    samples: List[Sample] = []
    for e in m:
        image_id = e.get("image_id") or os.path.splitext(os.path.basename(e["original_rel_path"]))[0]
        class_name = e.get("class_name") or e.get("shape_name") or e["wnid"]
        wnid = e.get("wnid") or e.get("shape_wnid")
        if wnid is None:
            raise KeyError("Original manifest entry missing wnid/shape_wnid")
        img_path = os.path.join(val_root, e["original_rel_path"])

        samples.append(
            Sample(
                sample_id=image_id,
                source_image_id=image_id,
                class_name=class_name,
                wnid=wnid,
                img_path=img_path,
                mode="original",
                background_name="original",
                output_rel_path=e.get("original_rel_path", ""),
                content_rel_path=e.get("original_rel_path", ""),
                condition_key=(class_name, "original"),
            )
        )
    return samples


def infer_background_name(entry: Dict[str, Any], output_rel_path: str) -> str:
    for key in ["background_name", "background", "bg_name"]:
        v = entry.get(key)
        if v:
            return str(v)

    # Try to infer from output path: images/<background>/<wnid>/file.jpg
    parts = Path(output_rel_path).parts
    if len(parts) >= 3 and parts[0] == "images":
        return parts[1]

    # Try to infer from filename
    stem = Path(output_rel_path).stem
    for candidate in ["shore", "openwater", "surface"]:
        if candidate in stem:
            return candidate

    return "unknown"


def build_samples_background_shift(gen_manifest_path: str, out_root: str) -> List[Sample]:
    with open(gen_manifest_path, "r") as f:
        m = json.load(f)

    samples: List[Sample] = []
    for e in m:
        output_rel_path = e.get("output_rel_path") or e.get("image_rel_path") or e.get("rel_path")
        if not output_rel_path:
            raise KeyError("Generated manifest entry missing output_rel_path/image_rel_path/rel_path.")

        source_image_id = (
            e.get("source_image_id")
            or e.get("image_id")
            or e.get("original_image_id")
            or os.path.splitext(os.path.basename(output_rel_path))[0]
        )
        sample_id = e.get("generated_image_id") or os.path.splitext(os.path.basename(output_rel_path))[0]
        class_name = e.get("class_name") or e.get("shape_name") or e.get("object_class_name")
        wnid = e.get("wnid") or e.get("shape_wnid") or e.get("object_wnid")
        if wnid is None:
            raise KeyError("Generated manifest entry missing wnid/shape_wnid/object_wnid.")
        if class_name is None:
            class_name = wnid

        background_name = infer_background_name(e, output_rel_path)
        img_path = os.path.join(out_root, output_rel_path)

        samples.append(
            Sample(
                sample_id=sample_id,
                source_image_id=source_image_id,
                class_name=class_name,
                wnid=wnid,
                img_path=img_path,
                mode="background_shift",
                background_name=background_name,
                output_rel_path=output_rel_path,
                content_rel_path=e.get("content_rel_path", ""),
                condition_key=(class_name, background_name),
            )
        )
    return samples


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
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


def find_rank(sorted_indices: Iterable[int], target_idx: int) -> int:
    for rank, idx in enumerate(sorted_indices, start=1):
        if int(idx) == int(target_idx):
            return rank
    return -1


# -----------------------------------------------------------------------------
# Manifest / path resolution
# -----------------------------------------------------------------------------
def resolve_default_gen_manifest(meta_dir: str) -> str:
    for name in DEFAULT_GEN_MANIFEST_CANDIDATES:
        p = os.path.join(meta_dir, name)
        if os.path.exists(p):
            return p
    # default to first candidate even if it does not exist yet
    return os.path.join(meta_dir, DEFAULT_GEN_MANIFEST_CANDIDATES[0])


# -----------------------------------------------------------------------------
# CSV + summary helpers
# -----------------------------------------------------------------------------
def write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_mode_summary(rows: List[Dict[str, Any]], mode: str) -> None:
    n = len(rows)
    if n == 0:
        print(f"[WARN] No rows to summarize for mode={mode}")
        return

    top1 = sum(int(r["correct_in_top1"]) for r in rows) / n
    top5 = sum(int(r["correct_in_top5"]) for r in rows) / n
    mean_prob = sum(float(r["prob_correct"]) for r in rows) / n
    mean_rank = sum(float(r["correct_rank"]) for r in rows) / n

    print(f"\n=== Overall summary ({mode}) ===")
    print(f"N: {n}")
    print(f"Top-1 accuracy: {100.0 * top1:.2f}%")
    print(f"Top-{max(5, max(int(r['topk']) for r in rows))} accuracy flag mean: {100.0 * top5:.2f}%")
    print(f"Mean correct-class probability: {mean_prob:.4f}")
    print(f"Mean correct-class rank: {mean_rank:.2f}")

    # By background
    by_bg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bg[r["background_name"]].append(r)

    print("\n=== By background ===")
    for bg, group in sorted(by_bg.items()):
        m = len(group)
        t1 = 100.0 * sum(int(r["correct_in_top1"]) for r in group) / max(1, m)
        t5 = 100.0 * sum(int(r["correct_in_top5"]) for r in group) / max(1, m)
        mp = sum(float(r["prob_correct"]) for r in group) / max(1, m)
        print(f"{bg:>12s} | N={m:3d} | top1={t1:6.2f}% | top5={t5:6.2f}% | prob_correct={mp:.4f}")


# -----------------------------------------------------------------------------
# Compare mode helpers
# -----------------------------------------------------------------------------
def load_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def compare_rows(original_rows: List[Dict[str, Any]], generated_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # original keyed by source/original image id
    orig_by_id: Dict[str, Dict[str, Any]] = {}
    for r in original_rows:
        key = r.get("source_image_id") or r.get("image_id")
        if key:
            orig_by_id[key] = r

    paired_rows: List[Dict[str, Any]] = []
    unmatched_generated: List[Dict[str, Any]] = []

    for g in generated_rows:
        key = g.get("source_image_id") or g.get("image_id")
        o = orig_by_id.get(key)
        if o is None:
            unmatched_generated.append(g)
            continue

        row = {
            "source_image_id": key,
            "generated_image_id": g.get("image_id", ""),
            "class_name": g.get("class_name") or o.get("class_name", ""),
            "wnid": g.get("wnid") or o.get("wnid", ""),
            "background_name": g.get("background_name", "unknown"),
            "original_pred_label": o.get("pred_label", ""),
            "generated_pred_label": g.get("pred_label", ""),
            "original_correct_in_top1": int(o.get("correct_in_top1", 0)),
            "generated_correct_in_top1": int(g.get("correct_in_top1", 0)),
            "original_correct_in_top5": int(o.get("correct_in_top5", 0)),
            "generated_correct_in_top5": int(g.get("correct_in_top5", 0)),
            "original_correct_rank": int(o.get("correct_rank", -1)),
            "generated_correct_rank": int(g.get("correct_rank", -1)),
            "original_prob_correct": float(o.get("prob_correct", 0.0)),
            "generated_prob_correct": float(g.get("prob_correct", 0.0)),
            "delta_prob_correct": float(g.get("prob_correct", 0.0)) - float(o.get("prob_correct", 0.0)),
            "original_logit_correct": float(o.get("logit_correct", 0.0)),
            "generated_logit_correct": float(g.get("logit_correct", 0.0)),
            "delta_logit_correct": float(g.get("logit_correct", 0.0)) - float(o.get("logit_correct", 0.0)),
            "pred_changed_vs_original": int(g.get("pred_wnid", "") != o.get("pred_wnid", "")),
            "generated_img_path": g.get("img_path", ""),
            "original_img_path": o.get("img_path", ""),
        }
        paired_rows.append(row)

    return paired_rows, unmatched_generated


def summarize_compare_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    by_bg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bg[r["background_name"]].append(r)

    for bg, group in sorted(by_bg.items()):
        n = len(group)
        summary_rows.append({
            "group": "background",
            "name": bg,
            "n": n,
            "original_top1_acc": sum(int(r["original_correct_in_top1"]) for r in group) / max(1, n),
            "generated_top1_acc": sum(int(r["generated_correct_in_top1"]) for r in group) / max(1, n),
            "original_top5_acc": sum(int(r["original_correct_in_top5"]) for r in group) / max(1, n),
            "generated_top5_acc": sum(int(r["generated_correct_in_top5"]) for r in group) / max(1, n),
            "mean_delta_prob_correct": sum(float(r["delta_prob_correct"]) for r in group) / max(1, n),
            "mean_delta_logit_correct": sum(float(r["delta_logit_correct"]) for r in group) / max(1, n),
            "pred_changed_rate": sum(int(r["pred_changed_vs_original"]) for r in group) / max(1, n),
        })

    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)

    for class_name, group in sorted(by_class.items()):
        n = len(group)
        summary_rows.append({
            "group": "class",
            "name": class_name,
            "n": n,
            "original_top1_acc": sum(int(r["original_correct_in_top1"]) for r in group) / max(1, n),
            "generated_top1_acc": sum(int(r["generated_correct_in_top1"]) for r in group) / max(1, n),
            "original_top5_acc": sum(int(r["original_correct_in_top5"]) for r in group) / max(1, n),
            "generated_top5_acc": sum(int(r["generated_correct_in_top5"]) for r in group) / max(1, n),
            "mean_delta_prob_correct": sum(float(r["delta_prob_correct"]) for r in group) / max(1, n),
            "mean_delta_logit_correct": sum(float(r["delta_logit_correct"]) for r in group) / max(1, n),
            "pred_changed_rate": sum(int(r["pred_changed_vs_original"]) for r in group) / max(1, n),
        })

    return summary_rows


# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------
def evaluate_samples(
    samples: List[Sample],
    model_name: str,
    device: torch.device,
    batch_size: int,
    topk: int,
    wnid_to_idx: Dict[str, int],
    idx_to_wnid: Dict[int, str],
    idx_to_label: Dict[int, str],
) -> List[Dict[str, Any]]:
    print(f"[INFO] Loading timm model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)

    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg, is_training=False)

    rows: List[Dict[str, Any]] = []
    counts_top1 = Counter()
    counts_top5 = Counter()
    counts_by_condition_top1 = defaultdict(Counter)

    with torch.no_grad():
        for batch in iter_batches(samples, transform, batch_size):
            batch_samples, batch_tensors = zip(*batch)
            x = torch.stack(batch_tensors, dim=0).to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            k = min(int(topk), probs.shape[1])
            topk_prob, topk_idx = probs.topk(k, dim=1)
            top1_idx = topk_idx[:, 0]
            top1_prob = topk_prob[:, 0]
            top1_logit = logits.gather(1, top1_idx.view(-1, 1)).squeeze(1)
            full_sorted_idx = probs.argsort(dim=1, descending=True)

            for i, s in enumerate(batch_samples):
                correct_idx = wnid_to_idx[s.wnid]
                pred_i = int(top1_idx[i].item())
                pred_wnid = idx_to_wnid[pred_i]
                pred_label = idx_to_label[pred_i]

                topk_set = set(int(v) for v in topk_idx[i].tolist())
                correct_in_top1 = int(pred_i == correct_idx)
                correct_in_top5 = int(correct_idx in topk_set)
                correct_rank = find_rank(full_sorted_idx[i].tolist(), correct_idx)

                logit_correct = float(logits[i, correct_idx].item())
                prob_correct = float(probs[i, correct_idx].item())
                prob_t1 = float(top1_prob[i].item())

                decision1 = "correct" if correct_in_top1 else "other"
                decisionk = "correct" if correct_in_top5 else "other"

                row = {
                    "mode": s.mode,
                    "image_id": s.sample_id,
                    "source_image_id": s.source_image_id,
                    "class_name": s.class_name,
                    "wnid": s.wnid,
                    "background_name": s.background_name,
                    "correct_in_top1": correct_in_top1,
                    "correct_in_top5": correct_in_top5,
                    "correct_rank": correct_rank,
                    "decision_top1": decision1,
                    "decision_topk": decisionk,
                    "pred_idx": pred_i,
                    "pred_wnid": pred_wnid,
                    "pred_label": pred_label,
                    "correct_idx": correct_idx,
                    "correct_label": idx_to_label[correct_idx],
                    "logit_correct": logit_correct,
                    "prob_correct": prob_correct,
                    "prob_top1": prob_t1,
                    "top1_logit_minus_correct_logit": float((top1_logit[i] - logits[i, correct_idx]).item()),
                    "topk": int(k),
                    "img_path": s.img_path,
                    "output_rel_path": s.output_rel_path,
                    "content_rel_path": s.content_rel_path,
                }
                rows.append(row)

                counts_top1[decision1] += 1
                counts_top5[decisionk] += 1
                counts_by_condition_top1[s.condition_key][decision1] += 1

    print("\n=== Overall decision rates (Top-1) ===")
    total = max(1, len(rows))
    for k in ["correct", "other"]:
        print(f"{k:7s}: {counts_top1[k]:5d} ({100.0 * counts_top1[k] / total:5.1f}%)")

    print("\n=== By condition (Top-1) ===")
    for cond, c in sorted(counts_by_condition_top1.items()):
        n = sum(c.values())
        corr = 100.0 * c["correct"] / max(1, n)
        oth = 100.0 * c["other"] / max(1, n)
        print(f"{cond[0]:>18s} + {cond[1]:>12s} | N={n:3d} | correct={corr:5.1f}% other={oth:5.1f}%")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Experiment YAML.")
    ap.add_argument("--mode", required=True, choices=["original", "background_shift", "compare"],
                    help="Evaluate original images, generated background-shift images, or compare both CSVs.")
    ap.add_argument("--model", default=None,
                    help="Override evaluation.model. Examples: resnet18.a1_in1k, resnet50.a1_in1k, vit_tiny_patch16_224")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Override evaluation.device.")
    ap.add_argument("--batch-size", type=int, default=None, help="Override evaluation.batch_size.")
    ap.add_argument("--topk", type=int, default=None, help="Override evaluation.topk.")
    ap.add_argument("--csv-out", default=None, help="Override output CSV path.")
    ap.add_argument("--class-index-json", default=None, help="Override imagenet_class_index.json path.")
    ap.add_argument("--gen-manifest", default=None,
                    help="(background_shift mode) override generated manifest path.")
    ap.add_argument("--original-csv", default=None,
                    help="(compare mode) path to evaluated original CSV. Default inferred from config/model.")
    ap.add_argument("--generated-csv", default=None,
                    help="(compare mode) path to evaluated background-shift CSV. Default inferred from config/model.")
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

    class_index_json = args.class_index_json or ev.get("class_index_json") or os.path.join(meta_dir, "imagenet_class_index.json")

    model_tag = model_name.replace(".", "_")
    gen_manifest = args.gen_manifest or ev.get("gen_manifest") or resolve_default_gen_manifest(meta_dir)
    default_original_csv = os.path.join(meta_dir, f"eval_original_{model_tag}_top{topk}.csv")
    default_generated_csv = os.path.join(meta_dir, f"eval_background_shift_{model_tag}_top{topk}.csv")
    default_compare_csv = os.path.join(meta_dir, f"eval_compare_background_shift_{model_tag}_top{topk}.csv")
    default_compare_summary_csv = os.path.join(meta_dir, f"eval_compare_background_shift_{model_tag}_top{topk}_summary.csv")

    device = torch.device("cuda" if (device_req == "cuda" and torch.cuda.is_available()) else "cpu")

    print("[INFO] Device:", device)
    print("[INFO] Mode:", args.mode)
    print("[INFO] Experiment out_root:", out_root)
    print("[INFO] Model:", model_name)
    print("[INFO] Batch_size:", batch_size, "topk:", topk)

    if args.mode == "compare":
        original_csv = args.original_csv or ev.get("original_csv") or default_original_csv
        generated_csv = args.generated_csv or ev.get("generated_csv") or default_generated_csv
        compare_csv = args.csv_out or ev.get("compare_csv") or default_compare_csv
        compare_summary_csv = ev.get("compare_summary_csv") or default_compare_summary_csv

        print("[INFO] original_csv:", original_csv)
        print("[INFO] generated_csv:", generated_csv)
        print("[INFO] compare_csv:", compare_csv)
        print("[INFO] compare_summary_csv:", compare_summary_csv)

        if not os.path.exists(original_csv):
            raise FileNotFoundError(f"Original CSV not found: {original_csv}")
        if not os.path.exists(generated_csv):
            raise FileNotFoundError(f"Generated CSV not found: {generated_csv}")

        original_rows = load_csv_rows(original_csv)
        generated_rows = load_csv_rows(generated_csv)
        paired_rows, unmatched_generated = compare_rows(original_rows, generated_rows)

        compare_header = [
            "source_image_id",
            "generated_image_id",
            "class_name",
            "wnid",
            "background_name",
            "original_pred_label",
            "generated_pred_label",
            "original_correct_in_top1",
            "generated_correct_in_top1",
            "original_correct_in_top5",
            "generated_correct_in_top5",
            "original_correct_rank",
            "generated_correct_rank",
            "original_prob_correct",
            "generated_prob_correct",
            "delta_prob_correct",
            "original_logit_correct",
            "generated_logit_correct",
            "delta_logit_correct",
            "pred_changed_vs_original",
            "generated_img_path",
            "original_img_path",
        ]
        write_csv(compare_csv, paired_rows, compare_header)
        print(f"[INFO] Wrote compare CSV: {compare_csv}")

        summary_rows = summarize_compare_rows(paired_rows)
        summary_header = [
            "group",
            "name",
            "n",
            "original_top1_acc",
            "generated_top1_acc",
            "original_top5_acc",
            "generated_top5_acc",
            "mean_delta_prob_correct",
            "mean_delta_logit_correct",
            "pred_changed_rate",
        ]
        write_csv(compare_summary_csv, summary_rows, summary_header)
        print(f"[INFO] Wrote compare summary CSV: {compare_summary_csv}")
        print(f"[INFO] Paired rows: {len(paired_rows)} | unmatched generated rows: {len(unmatched_generated)}")
        return

    ensure_imagenet_class_index(class_index_json)
    idx_to_wnid, wnid_to_idx, idx_to_label = load_class_index_json(class_index_json)

    if args.mode == "original":
        samples = build_samples_original(content_manifest, val_root)
        csv_out = args.csv_out or ev.get("csv_out") or default_original_csv
    else:
        if gen_manifest is None or not os.path.exists(gen_manifest):
            raise FileNotFoundError(f"Generated manifest not found: {gen_manifest}")
        samples = build_samples_background_shift(gen_manifest, out_root)
        csv_out = args.csv_out or ev.get("csv_out") or default_generated_csv

    print(f"[INFO] Loaded {len(samples)} samples")
    print(f"[INFO] csv_out: {csv_out}")
    if args.mode == "background_shift":
        print(f"[INFO] gen_manifest: {gen_manifest}")

    missing_wnids = sorted({s.wnid for s in samples if s.wnid not in wnid_to_idx})
    if missing_wnids:
        raise KeyError(f"Some WNIDs are missing from imagenet_class_index.json: {missing_wnids[:25]}")

    rows = evaluate_samples(
        samples=samples,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        topk=topk,
        wnid_to_idx=wnid_to_idx,
        idx_to_wnid=idx_to_wnid,
        idx_to_label=idx_to_label,
    )

    header = [
        "mode",
        "image_id",
        "source_image_id",
        "class_name",
        "wnid",
        "background_name",
        "correct_in_top1",
        "correct_in_top5",
        "correct_rank",
        "decision_top1",
        "decision_topk",
        "pred_idx",
        "pred_wnid",
        "pred_label",
        "correct_idx",
        "correct_label",
        "logit_correct",
        "prob_correct",
        "prob_top1",
        "top1_logit_minus_correct_logit",
        "topk",
        "img_path",
        "output_rel_path",
        "content_rel_path",
    ]
    write_csv(csv_out, rows, header)
    print(f"\n[INFO] Wrote CSV: {csv_out}")
    print_mode_summary(rows, args.mode)


if __name__ == "__main__":
    main()

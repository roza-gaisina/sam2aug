#!/usr/bin/env python3
"""
Unified evaluator for:
  --mode geirhos  (generated images)
  --mode original (original images)

Config-driven (supports inherits like your generator):
- out_root = paths.outputs.root / experiment.name
- val_root = paths.imagenet.val_root
- content.manifest is the curated original set (always)
- cue-conflict gen manifest is read from out_root/meta/geirhos_gen_manifest.json

Outputs:
- CSV with top1/top5 decisions, parent-category metrics, and shape/texture logits & probabilities.
- Summary JSON next to the CSV with aggregate metrics.

Run:
python evaluate_geirhos.py --config configs/geirhos_texture_only.yaml --mode geirhos --model resnet18.a1_in1k
python evaluate_geirhos.py --config configs/geirhos_texture_plus_edges.yaml --mode geirhos --model resnet18.a1_in1k
python evaluate_geirhos.py --config configs/geirhos_texture_nst.yaml --mode geirhos --model resnet18.a1_in1k
python evaluate_geirhos.py --config configs/geirhos_texture_adain.yaml --mode geirhos --model resnet18.a1_in1k
python evaluate_geirhos.py --config configs/base.yaml --mode original --model resnet18.a1_in1k
"""

import argparse
import csv
import json
import math
import os
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import timm
from timm.data import create_transform, resolve_data_config

# YAML parsing
try:
    import yaml
except ImportError as e:
    raise ImportError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from e


DEFAULT_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

# ----------------------------
# Custom semantic parent categories for additional robustness analysis
# ----------------------------
PARENT_CATEGORIES: Dict[str, List[str]] = {
    "airplane": [
        "n02690373",  # airliner
    ],
    "bear": [
        "n02132136",  # brown_bear
        "n02133161",  # American_black_bear
        "n02134084",  # ice_bear
        "n02134418",  # sloth_bear
    ],
    "bicycle": [
        "n02835271",  # bicycle-built-for-two
        "n03792782",  # mountain_bike
    ],
    "bird": [
        "n01795545", "n01796340", "n01797886", "n01798484", "n01806143",
        "n01806567", "n01807496", "n01817953", "n01818515", "n01819313",
        "n01820546", "n01824575", "n01828970", "n01829413", "n01833805",
        "n01843065", "n01843383", "n01847000", "n01855032", "n01855672",
        "n01860187", "n02002556", "n02002724", "n02006656", "n02007558",
        "n02009229", "n02009912", "n02011460", "n02012849", "n02013706",
        "n02017213", "n02018207", "n02018795", "n02025239", "n02027492",
        "n02028035", "n02033041", "n02037110", "n02051845", "n02056570",
        "n02058221", "n01514668", "n01514859", "n01518878", "n01530575",
        "n01531178", "n01532829", "n01534433", "n01537544", "n01558993",
        "n01560419", "n01580077", "n01582220", "n01592084", "n01601694",
        "n01608432", "n01614925", "n01616318", "n01622779",
    ],
    "boat": [
        "n02951358", "n03344393", "n03447447", "n03662601", "n04273569",
        "n04612504", "n02981792", "n04483307", "n03095699", "n03673027",
        "n02687172", "n04347754", "n04147183",
    ],
    "bottle": [
        "n02823428", "n02877765", "n03983396", "n03937543", "n04591713",
        "n04557648", "n04560804",
    ],
    "car": [
        "n02814533", "n02930766", "n03100240", "n03594945", "n03670208",
        "n03770679", "n03777568", "n04037443", "n04285008",
    ],
    "cat": [
        "n02123045", "n02123159", "n02123394", "n02123597", "n02124075", "n02125311",
    ],
    "chair": [
        "n02791124", "n03376595", "n04099969", "n04429376",
    ],
    "clock": [
        "n02708093", "n03196217", "n04548280",
    ],
    "dog": [
        "n02085620", "n02085782", "n02085936", "n02086079", "n02086240",
        "n02086646", "n02086910", "n02087046", "n02087394", "n02088094",
        "n02088238", "n02088364", "n02088466", "n02088632", "n02089078",
        "n02089867", "n02089973", "n02090379", "n02090622", "n02090721",
        "n02091032", "n02091134", "n02091244", "n02091467", "n02091635",
        "n02091831", "n02092002", "n02092339", "n02093256", "n02093428",
        "n02093647", "n02093754", "n02093859", "n02093991", "n02094114",
        "n02094258", "n02094433", "n02095314", "n02095570", "n02095889",
        "n02096051", "n02096177", "n02096294", "n02096437", "n02096585",
        "n02097047", "n02097130", "n02097209", "n02097298", "n02097474",
        "n02097658", "n02098105", "n02098286", "n02098413", "n02099267",
        "n02099429", "n02099601", "n02099712", "n02099849", "n02100236",
        "n02100583", "n02100735", "n02100877", "n02101006", "n02101388",
        "n02101556", "n02102040", "n02102177", "n02102318", "n02102480",
        "n02102973", "n02104029", "n02104365", "n02105056", "n02105162",
        "n02105251", "n02105412", "n02105505", "n02105641", "n02105855",
        "n02106030", "n02106166", "n02106382", "n02106550", "n02106662",
        "n02107142", "n02107312", "n02107574", "n02107683", "n02107908",
        "n02108000", "n02108089", "n02108422", "n02108551", "n02108915",
        "n02109047", "n02109525", "n02109961", "n02110063", "n02110185",
        "n02110341", "n02110627", "n02110806", "n02110958", "n02111129",
        "n02111277", "n02111500", "n02111889", "n02112018", "n02112137",
        "n02112350", "n02112706", "n02113023", "n02113186", "n02113624",
        "n02113712", "n02113799", "n02113978",
    ],
    "elephant": [
        "n02504013", "n02504458",
    ],
    "keyboard": [
        "n03085013", "n04505470",
    ],
    "knife": [
        "n03041632",
    ],
    "truck": [
        "n03345487", "n03417042", "n03796401", "n03930630",
        "n03977966", "n04461696", "n04467665",
    ],
}


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

    idx_to_wnid = {}
    idx_to_label = {}
    for k, (wnid, label) in data.items():
        i = int(k)
        idx_to_wnid[i] = wnid
        idx_to_label[i] = label

    wnid_to_idx = {wnid: i for i, wnid in idx_to_wnid.items()}
    return idx_to_wnid, wnid_to_idx, idx_to_label


# ----------------------------
# Semantic parent helpers
# ----------------------------
def build_wnid_to_parent(parent_categories: Dict[str, List[str]]) -> Dict[str, str]:
    wnid_to_parent: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)
    for parent_name, wnids in parent_categories.items():
        for wnid in wnids:
            if wnid in wnid_to_parent and wnid_to_parent[wnid] != parent_name:
                duplicates[wnid].append(parent_name)
            else:
                wnid_to_parent[wnid] = parent_name
    if duplicates:
        raise ValueError(f"Some WNIDs appear in multiple parent categories: {dict(duplicates)}")
    return wnid_to_parent


def get_parent_name(wnid: str, wnid_to_parent: Dict[str, str], fallback_to_wnid: bool = True) -> Optional[str]:
    parent = wnid_to_parent.get(wnid)
    if parent is not None:
        return parent
    return wnid if fallback_to_wnid else None


def get_parent_members(
    parent_name: Optional[str],
    parent_categories: Dict[str, List[str]],
    wnid_to_idx: Dict[str, int],
    fallback_wnid: Optional[str] = None,
) -> List[int]:
    if parent_name is None:
        return []
    if parent_name in parent_categories:
        members = [wnid_to_idx[w] for w in parent_categories[parent_name] if w in wnid_to_idx]
        if members:
            return members
    if fallback_wnid is not None and fallback_wnid in wnid_to_idx:
        return [wnid_to_idx[fallback_wnid]]
    return []


# ----------------------------
# AUC helpers
# ----------------------------
def binary_auc_ovr(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Computes binary ROC AUC from scores using the Mann-Whitney formulation.
    Returns None if the class has only positives or only negatives.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    n = len(y_score)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def macro_ovr_auc(class_scores: np.ndarray, class_wnids: List[str], true_wnids: List[str]) -> Dict[str, Any]:
    per_class_auc: Dict[str, float] = {}
    true_arr = np.array(true_wnids)
    for j, wnid in enumerate(class_wnids):
        y_true = (true_arr == wnid).astype(np.int64)
        auc = binary_auc_ovr(y_true, class_scores[:, j])
        if auc is not None:
            per_class_auc[wnid] = auc

    macro_auc = None
    if per_class_auc:
        macro_auc = float(np.mean(list(per_class_auc.values())))

    return {
        "macro_ovr_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "num_auc_classes": len(per_class_auc),
    }


# ----------------------------
# Data
# ----------------------------
def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class Sample:
    image_id: str
    shape_name: str
    shape_wnid: str
    texture_wnid: str
    texture_name: str
    img_path: str
    condition_key: Tuple[str, str]


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


def build_samples_original(content_manifest_path: str, val_root: str) -> List[Sample]:
    with open(content_manifest_path, "r") as f:
        m = json.load(f)

    samples: List[Sample] = []
    for e in m:
        image_id = e.get("image_id") or os.path.splitext(os.path.basename(e["original_rel_path"]))[0]
        shape_name = e["class_name"]
        shape_wnid = e["wnid"]
        img_path = os.path.join(val_root, e["original_rel_path"])

        samples.append(
            Sample(
                image_id=image_id,
                shape_name=shape_name,
                shape_wnid=shape_wnid,
                texture_wnid=shape_wnid,
                texture_name=shape_name,
                img_path=img_path,
                condition_key=(shape_name, shape_name),
            )
        )
    return samples


def build_samples_geirhos(gen_manifest_path: str, out_root: str, texture_name_map: Dict[str, str]) -> List[Sample]:
    with open(gen_manifest_path, "r") as f:
        m = json.load(f)

    samples: List[Sample] = []
    for e in m:
        image_id = e.get("image_id") or os.path.splitext(os.path.basename(e["output_rel_path"]))[0]
        shape_name = e["shape_name"]
        shape_wnid = e.get("shape_wnid") or e.get("wnid")
        texture_wnid = e.get("texture_wnid")

        if shape_wnid is None:
            raise KeyError("Gen manifest entry missing shape_wnid/wnid.")
        if texture_wnid is None:
            raise KeyError("Gen manifest entry missing texture_wnid.")

        texture_name = texture_name_map.get(texture_wnid, texture_wnid)
        img_path = os.path.join(out_root, e["output_rel_path"])

        samples.append(
            Sample(
                image_id=image_id,
                shape_name=shape_name,
                shape_wnid=shape_wnid,
                texture_wnid=texture_wnid,
                texture_name=texture_name,
                img_path=img_path,
                condition_key=(shape_name, texture_name),
            )
        )
    return samples


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Experiment YAML (e.g. geirhos16.yaml).")
    ap.add_argument("--mode", required=True, choices=["geirhos", "original"], help="Evaluate generated cue-conflict images or original images.")
    ap.add_argument("--model", default=None, help="Override evaluation.model (e.g. resnet18.a1_in1k).")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Override evaluation.device.")
    ap.add_argument("--batch-size", type=int, default=None, help="Override evaluation.batch_size.")
    ap.add_argument("--topk", type=int, default=None, help="Override evaluation.topk.")
    ap.add_argument("--csv-out", default=None, help="Override output CSV path.")
    ap.add_argument("--class-index-json", default=None, help="Override imagenet_class_index.json path.")
    ap.add_argument("--gen-manifest", default=None, help="(geirhos mode) override gen manifest path. Default: <out_root>/meta/geirhos_gen_manifest.json")
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

    if args.mode == "geirhos":
        gen_manifest = args.gen_manifest or ev.get("gen_manifest") or os.path.join(meta_dir, "geirhos_gen_manifest.json")
        default_csv = os.path.join(meta_dir, f"eval_{exp_name}_{model_name.replace('.', '_')}_top{topk}.csv")
    else:
        gen_manifest = None
        default_csv = os.path.join(meta_dir, f"eval_original_{model_name.replace('.', '_')}_top{topk}.csv")

    csv_out = args.csv_out or ev.get("csv_out") or default_csv
    summary_out = os.path.splitext(csv_out)[0] + "_summary.json"

    tex_cfg = cfg.get("textures", {})
    texture_name_map = tex_cfg.get("texture_names", {}) or {}
    if not isinstance(texture_name_map, dict):
        raise ValueError("textures.texture_names must be a dict mapping wnid -> human-readable name")

    device = torch.device("cuda" if (device_req == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Mode:", args.mode)
    print("[INFO] Experiment out_root:", out_root)
    print("[INFO] Model:", model_name)
    print("[INFO] Batch_size:", batch_size, "topk:", topk)
    if args.mode == "geirhos":
        print("[INFO] gen_manifest:", gen_manifest)
    print("[INFO] csv_out:", csv_out)
    print("[INFO] summary_out:", summary_out)

    ensure_imagenet_class_index(class_index_json)
    idx_to_wnid, wnid_to_idx, idx_to_label = load_class_index_json(class_index_json)

    if args.mode == "original":
        samples = build_samples_original(content_manifest, val_root)
    else:
        if gen_manifest is None or not os.path.exists(gen_manifest):
            raise FileNotFoundError(f"Gen manifest not found: {gen_manifest}")
        samples = build_samples_geirhos(gen_manifest, out_root, texture_name_map)

    print(f"[INFO] Loaded {len(samples)} samples")

    missing_wnids = set()
    for s in samples:
        if s.shape_wnid not in wnid_to_idx:
            missing_wnids.add(s.shape_wnid)
        if s.texture_wnid not in wnid_to_idx:
            missing_wnids.add(s.texture_wnid)
    if missing_wnids:
        raise KeyError(f"Some WNIDs are missing from imagenet_class_index.json: {sorted(missing_wnids)[:25]}")

    wnid_to_parent = build_wnid_to_parent(PARENT_CATEGORIES)
    sample_shape_wnids = sorted({s.shape_wnid for s in samples})
    missing_shape_parents = sorted([w for w in sample_shape_wnids if w not in wnid_to_parent])
    if missing_shape_parents:
        print(f"[WARN] {len(missing_shape_parents)} shape WNIDs are not in PARENT_CATEGORIES; falling back to exact-class parent for them.")

    print(f"[INFO] Loading timm model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)

    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg, is_training=False)
    
    rows = []
    counts_top1 = Counter()
    counts_top5 = Counter()
    counts_by_condition_top1 = defaultdict(Counter)
    counts_by_condition_top5 = defaultdict(Counter)

    parent_top1_hits = 0
    parent_top5_hits = 0
    parent_prob_mass_values: List[float] = []

    auc_eval_wnids = sorted({s.shape_wnid for s in samples}) if args.mode == "original" else []
    auc_eval_indices = [wnid_to_idx[w] for w in auc_eval_wnids]
    auc_true_wnids: List[str] = []
    auc_score_chunks: List[np.ndarray] = []

    header = [
        "mode",
        "image_id",
        "shape_name",
        "shape_wnid",
        "texture_name",
        "texture_wnid",
        "shape_parent",
        "pred_parent",
        "decision_top1",
        "decision_top5",
        "shape_in_top5",
        "texture_in_top5",
        "parent_top1_match",
        "parent_top5_match",
        "parent_prob_mass",
        "pred_idx",
        "pred_wnid",
        "pred_label",
        "shape_idx",
        "texture_idx",
        "logit_shape",
        "logit_texture",
        "prob_shape",
        "prob_texture",
        "prob_top1",
        "top1_logit_minus_shape_logit",
        "top1_logit_minus_texture_logit",
        "img_path",
    ]

    with torch.no_grad():
        for batch in iter_batches(samples, transform, batch_size):
            batch_samples, batch_tensors = zip(*batch)
            x = torch.stack(batch_tensors, dim=0).to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            if args.mode == "original" and auc_eval_indices:
                auc_true_wnids.extend([s.shape_wnid for s in batch_samples])
                auc_score_chunks.append(probs[:, auc_eval_indices].detach().cpu().numpy())

            topk_prob, topk_idx = probs.topk(topk, dim=1)
            top1_idx = topk_idx[:, 0]
            top1_prob = topk_prob[:, 0]
            top1_logit = logits.gather(1, top1_idx.view(-1, 1)).squeeze(1)

            for i, s in enumerate(batch_samples):
                shape_idx = wnid_to_idx[s.shape_wnid]
                texture_idx = wnid_to_idx[s.texture_wnid]

                pred_i = int(top1_idx[i].item())
                pred_wnid = idx_to_wnid[pred_i]
                pred_label = idx_to_label[pred_i]

                if pred_i == shape_idx:
                    decision1 = "shape"
                elif pred_i == texture_idx:
                    decision1 = "texture"
                else:
                    decision1 = "other"

                topk_list = topk_idx[i].tolist()
                topk_set = set(topk_list)
                shape_in = shape_idx in topk_set
                tex_in = texture_idx in topk_set

                if shape_in and tex_in:
                    decisionk = "both"
                elif tex_in:
                    decisionk = "texture"
                elif shape_in:
                    decisionk = "shape"
                else:
                    decisionk = "other"

                shape_parent = get_parent_name(s.shape_wnid, wnid_to_parent, fallback_to_wnid=True)
                pred_parent = get_parent_name(pred_wnid, wnid_to_parent, fallback_to_wnid=True)
                parent_member_indices = get_parent_members(shape_parent, PARENT_CATEGORIES, wnid_to_idx, fallback_wnid=s.shape_wnid)
                parent_top1_match = int(pred_parent == shape_parent)
                parent_top5_match = int(any(idx_to_wnid[idx] in set([w for w in PARENT_CATEGORIES.get(shape_parent, [])]) for idx in topk_list))
                if shape_parent not in PARENT_CATEGORIES:
                    parent_top5_match = int(shape_idx in topk_set)
                parent_prob_mass = float(probs[i, parent_member_indices].sum().item()) if parent_member_indices else 0.0

                logit_shape = float(logits[i, shape_idx].item())
                logit_tex = float(logits[i, texture_idx].item())
                prob_shape = float(probs[i, shape_idx].item())
                prob_tex = float(probs[i, texture_idx].item())
                prob_t1 = float(top1_prob[i].item())

                rows.append({
                    "mode": args.mode,
                    "image_id": s.image_id,
                    "shape_name": s.shape_name,
                    "shape_wnid": s.shape_wnid,
                    "texture_name": s.texture_name,
                    "texture_wnid": s.texture_wnid,
                    "shape_parent": shape_parent,
                    "pred_parent": pred_parent,
                    "decision_top1": decision1,
                    "decision_top5": decisionk,
                    "shape_in_top5": int(shape_in),
                    "texture_in_top5": int(tex_in),
                    "parent_top1_match": parent_top1_match,
                    "parent_top5_match": parent_top5_match,
                    "parent_prob_mass": parent_prob_mass,
                    "pred_idx": pred_i,
                    "pred_wnid": pred_wnid,
                    "pred_label": pred_label,
                    "shape_idx": shape_idx,
                    "texture_idx": texture_idx,
                    "logit_shape": logit_shape,
                    "logit_texture": logit_tex,
                    "prob_shape": prob_shape,
                    "prob_texture": prob_tex,
                    "prob_top1": prob_t1,
                    "top1_logit_minus_shape_logit": float((top1_logit[i] - logits[i, shape_idx]).item()),
                    "top1_logit_minus_texture_logit": float((top1_logit[i] - logits[i, texture_idx]).item()),
                    "img_path": s.img_path,
                })

                counts_top1[decision1] += 1
                counts_top5[decisionk] += 1
                counts_by_condition_top1[s.condition_key][decision1] += 1
                counts_by_condition_top5[s.condition_key][decisionk] += 1
                parent_top1_hits += parent_top1_match
                parent_top5_hits += parent_top5_match
                parent_prob_mass_values.append(parent_prob_mass)

    n = len(rows)

    def pct(x: float) -> float:
        return 100.0 * x / max(1, n)

    print("\n=== Overall decision rates (Top-1) ===")
    for k in ["shape", "texture", "other"]:
        print(f"{k:7s}: {counts_top1[k]:5d} ({pct(counts_top1[k]):5.1f}%)")

    print("\n=== Overall decision rates (Top-5 label) ===")
    for k in ["shape", "texture", "both", "other"]:
        print(f"{k:7s}: {counts_top5[k]:5d} ({pct(counts_top5[k]):5.1f}%)")

    print("\n=== Semantic robustness metrics ===")
    print(f"parent_top1_match : {parent_top1_hits:5d} ({pct(parent_top1_hits):5.1f}%)")
    print(f"parent_top5_match : {parent_top5_hits:5d} ({pct(parent_top5_hits):5.1f}%)")
    mean_parent_prob_mass = float(np.mean(parent_prob_mass_values)) if parent_prob_mass_values else 0.0
    print(f"parent_prob_mass  : {mean_parent_prob_mass:.4f} (mean)")

    print("\n=== By condition (Top-1) ===")
    for cond, c in sorted(counts_by_condition_top1.items()):
        total = sum(c.values())
        sh = 100.0 * c["shape"] / max(1, total)
        tx = 100.0 * c["texture"] / max(1, total)
        ot = 100.0 * c["other"] / max(1, total)
        print(f"{cond[0]:>12s} + {cond[1]:>12s} | N={total:3d} | shape={sh:5.1f}% texture={tx:5.1f}% other={ot:5.1f}%")

    auc_summary: Dict[str, Any] = {}
    if args.mode == "original" and auc_score_chunks:
        auc_scores = np.concatenate(auc_score_chunks, axis=0)
        auc_summary = macro_ovr_auc(auc_scores, auc_eval_wnids, auc_true_wnids)
        macro_auc = auc_summary.get("macro_ovr_auc")
        print("\n=== Original-image AUC ===")
        if macro_auc is None:
            print("macro_ovr_auc: unavailable (need at least one positive and one negative for each class)")
        else:
            print(f"macro_ovr_auc: {macro_auc:.4f} over {auc_summary['num_auc_classes']} classes")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "mode": args.mode,
        "model": model_name,
        "num_samples": n,
        "topk": topk,
        "decision_top1": {k: counts_top1[k] for k in ["shape", "texture", "other"]},
        "decision_top1_pct": {k: pct(counts_top1[k]) for k in ["shape", "texture", "other"]},
        "decision_top5": {k: counts_top5[k] for k in ["shape", "texture", "both", "other"]},
        "decision_top5_pct": {k: pct(counts_top5[k]) for k in ["shape", "texture", "both", "other"]},
        "semantic_robustness": {
            "parent_top1_match_count": parent_top1_hits,
            "parent_top1_match_pct": pct(parent_top1_hits),
            "parent_top5_match_count": parent_top5_hits,
            "parent_top5_match_pct": pct(parent_top5_hits),
            "parent_prob_mass_mean": mean_parent_prob_mass,
        },
        "original_image_auc": auc_summary if args.mode == "original" else {},
    }

    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[INFO] Wrote CSV:", csv_out)
    print("[INFO] Wrote summary JSON:", summary_out)


if __name__ == "__main__":
    main()

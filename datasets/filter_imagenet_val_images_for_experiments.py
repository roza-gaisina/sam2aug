#!/usr/bin/env python3
"""
Select ImageNet validation images for Geirhos-style experiments.

This script:
1. Reads ImageNet val images from class-folder structure.
2. Filters images to selected parent categories (via WNIDs).
3. Keeps only images with exactly one XML bounding box.
4. Optionally requires a minimum bbox area ratio.
5. Evaluates four models:
     - resnet18.a1_in1k
     - resnet50.a1_in1k
     - vit_tiny_patch16_224.augreg_in21k_ft_in1k
     - vit_base_patch16_224.augreg_in21k_ft_in1k
6. Keeps only images correctly classified by all four models.
7. Saves up to K images per WNID for manual inspection.
8. Writes CSV summaries.

Example:
python select_imagenet_consistent_shapes.py \
    --val-dir /path/to/imagenet/val \
    --annotation-dir /path/to/imagenet/val_xml \
    --output-dir /path/to/output/shape_candidates \
    --batch-size 32 \
    --max-images-per-class 10 \
    --min-bbox-area-ratio 0.20
"""

import argparse
import csv
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

import timm
from timm.data import create_transform, resolve_data_config
from tqdm.auto import tqdm


# ------------------------------------------------------------
# Parent categories
# ------------------------------------------------------------
PARENT_CATEGORIES = {
    "airplane": [
        "n02690373",
    ],
    "bear": [
        "n02132136",
        "n02133161",
        "n02134084",
        "n02134418",
    ],
    "bicycle": [
        "n02835271",
        "n03792782",
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
        "n02823428", "n02877765", "n03983396", "n03937543",
        "n04591713", "n04557648", "n04560804",
    ],
    "car": [
        "n02814533", "n02930766", "n03100240", "n03594945",
        "n03670208", "n03777568", "n04037443", "n04285008",
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
    "oven": [
        "n04111531",
    ],
    "truck": [
        "n03345487", "n03417042", "n03770679", "n03796401",
        "n03930630", "n03977966", "n04461696", "n04467665",
    ],
    "notebook": [
        "n03832673",
    ],
    "teapot": [
        "n04398044",
    ],
}

MODEL_NAMES = {
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_wnid_to_parent(parent_categories: Dict[str, List[str]]) -> Dict[str, str]:
    wnid_to_parent = {}
    duplicates = defaultdict(list)

    for parent, wnids in parent_categories.items():
        for wnid in wnids:
            if wnid in wnid_to_parent:
                duplicates[wnid].append(parent)
            else:
                wnid_to_parent[wnid] = parent

    if duplicates:
        print("[WARN] Overlapping WNIDs found:")
        for wnid, parents in duplicates.items():
            print(f"  {wnid}: existing={wnid_to_parent[wnid]}, also_in={parents}")

    return wnid_to_parent


def parse_imagenet_xml(xml_path: Path) -> Optional[dict]:
    if not xml_path.exists():
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_names = []
    bboxes = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))

        object_names.append(name)
        bboxes.append((xmin, ymin, xmax, ymax))

    return {
        "object_names": object_names,
        "bboxes": bboxes,
    }


def compute_bbox_area_ratio(img_path: Path, bbox) -> float:
    with Image.open(img_path) as img:
        w, h = img.size

    xmin, ymin, xmax, ymax = bbox
    bbox_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    img_area = w * h
    return bbox_area / img_area if img_area > 0 else 0.0


def prefilter_candidates(
    val_dir: Path,
    annotation_dir: Path,
    wnid_to_parent: Dict[str, str],
    min_bbox_area_ratio: float,
) -> pd.DataFrame:
    dataset = ImageFolder(val_dir)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    all_target_wnids = set(wnid_to_parent.keys())

    print(f"[INFO] Validation images: {len(dataset)}")
    print(f"[INFO] Validation classes: {len(dataset.class_to_idx)}")

    rows = []

    for img_path_str, target_idx in tqdm(dataset.samples, desc="Prefiltering candidates"):
        img_path = Path(img_path_str)
        wnid = idx_to_class[target_idx]

        if wnid not in all_target_wnids:
            continue

        ann = parse_imagenet_xml(annotation_dir / f"{img_path.stem}.xml")
        if ann is None:
            continue

        if len(ann["bboxes"]) != 1:
            continue

        bbox = ann["bboxes"][0]
        bbox_area_ratio = compute_bbox_area_ratio(img_path, bbox)

        if bbox_area_ratio < min_bbox_area_ratio:
            continue

        rows.append({
            "row_id": len(rows),
            "img_path": str(img_path),
            "file_name": img_path.name,
            "stem": img_path.stem,
            "wnid": wnid,
            "parent_category": wnid_to_parent[wnid],
            "target_idx": int(target_idx),
            "bbox_count": 1,
            "bbox_area_ratio": float(bbox_area_ratio),
            "object_names_xml": "|".join(ann["object_names"]),
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Candidates after bbox filtering: {len(df)}")
    return df


def evaluate_model_on_candidates(
    model_name: str,
    candidate_df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    print(f"[INFO] Loading timm model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)

    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg, is_training=False)

    results = []

    with torch.no_grad():
        for start in tqdm(range(0, len(candidate_df), batch_size), desc=f"Evaluating {model_name}"):
            batch_df = candidate_df.iloc[start:start + batch_size]

            images = []
            row_ids = []
            target_idxs = []

            for _, row in batch_df.iterrows():
                with Image.open(row["img_path"]) as img:
                    image = img.convert("RGB")
                images.append(transform(image))
                row_ids.append(int(row["row_id"]))
                target_idxs.append(int(row["target_idx"]))

            x = torch.stack(images, dim=0).to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            top1_idx = logits.argmax(dim=1)
            top1_prob = probs.gather(1, top1_idx.view(-1, 1)).squeeze(1)

            for j in range(len(row_ids)):
                pred_idx = int(top1_idx[j].item())
                results.append({
                    "row_id": row_ids[j],
                    "pred_idx": pred_idx,
                    "prob_top1": float(top1_prob[j].item()),
                    "is_correct": int(pred_idx == target_idxs[j]),
                })

    result_df = pd.DataFrame(results).sort_values("row_id").reset_index(drop=True)
    return result_df


def prepare_eval_df(df: pd.DataFrame, pred_col: str, prob_col: str, correct_col: str, model_label: str) -> pd.DataFrame:
    required_cols = {"row_id", "pred_idx", "prob_top1", "is_correct"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{model_label} missing columns: {missing}")

    if df["row_id"].duplicated().any():
        dupes = df[df["row_id"].duplicated()]["row_id"].tolist()[:10]
        raise ValueError(f"{model_label} duplicate row_id values: {dupes}")

    out = df.rename(columns={
        "pred_idx": pred_col,
        "prob_top1": prob_col,
        "is_correct": correct_col,
    }).copy()

    return out.set_index("row_id")[[pred_col, prob_col, correct_col]]


def merge_model_results(candidate_df: pd.DataFrame, eval_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    results_df = candidate_df.copy().set_index("row_id")

    for model_key, df in eval_tables.items():
        results_df = results_df.join(df, how="left")

    results_df = results_df.reset_index()

    required_prediction_cols = [
        "correct_r18", "correct_r50", "correct_vit_tiny", "correct_vit_base",
        "prob_r18", "prob_r50", "prob_vit_tiny", "prob_vit_base",
    ]

    missing_rows = results_df[required_prediction_cols].isna().any(axis=1).sum()
    if missing_rows > 0:
        bad = results_df[results_df[required_prediction_cols].isna().any(axis=1)].head(10)
        raise ValueError(
            "Some rows are missing predictions after joining.\n"
            f"Example bad rows:\n{bad[['row_id', 'img_path', 'wnid']].to_string(index=False)}"
        )

    results_df["all_correct"] = (
        results_df["correct_r18"].astype(int).astype(bool)
        & results_df["correct_r50"].astype(int).astype(bool)
        & results_df["correct_vit_tiny"].astype(int).astype(bool)
        & results_df["correct_vit_base"].astype(int).astype(bool)
    ).astype(int)

    results_df["mean_prob"] = results_df[
        ["prob_r18", "prob_r50", "prob_vit_tiny", "prob_vit_base"]
    ].mean(axis=1)

    return results_df


def save_selected_images(selected_df: pd.DataFrame, output_dir: Path) -> None:
    save_root = output_dir / "selected_images"
    save_root.mkdir(parents=True, exist_ok=True)

    for _, row in selected_df.iterrows():
        src = Path(row["img_path"])
        dst_dir = save_root / row["parent_category"] / row["wnid"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)

    print(f"[INFO] Saved selected images to: {save_root}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Wrote CSV: {path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True, help="ImageNet val root in class-folder structure.")
    parser.add_argument("--annotation-dir", required=True, help="Directory with ImageNet val XML annotations.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images-per-class", type=int, default=10)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.20)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--vit-base-name", default="vit_base_patch16_224.augreg_in21k_ft_in1k")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print("[INFO] Device:", device)
    print("[INFO] val_dir:", val_dir)
    print("[INFO] annotation_dir:", annotation_dir)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] batch_size:", args.batch_size)
    print("[INFO] max_images_per_class:", args.max_images_per_class)
    print("[INFO] min_bbox_area_ratio:", args.min_bbox_area_ratio)

    model_names = dict(MODEL_NAMES)
    model_names["vit_base"] = args.vit_base_name

    print("[INFO] Models:")
    for k, v in model_names.items():
        print(f"  {k}: {v}")

    wnid_to_parent = build_wnid_to_parent(PARENT_CATEGORIES)
    candidate_df = prefilter_candidates(
        val_dir=val_dir,
        annotation_dir=annotation_dir,
        wnid_to_parent=wnid_to_parent,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
    )

    if len(candidate_df) == 0:
        print("[WARN] No candidate images found after prefiltering.")
        return

    eval_r18 = evaluate_model_on_candidates(model_names["resnet18"], candidate_df, args.batch_size, device)
    eval_r50 = evaluate_model_on_candidates(model_names["resnet50"], candidate_df, args.batch_size, device)
    eval_vit_tiny = evaluate_model_on_candidates(model_names["vit_tiny"], candidate_df, args.batch_size, device)
    eval_vit_base = evaluate_model_on_candidates(model_names["vit_base"], candidate_df, args.batch_size, device)

    eval_tables = {
        "r18": prepare_eval_df(eval_r18, "pred_r18", "prob_r18", "correct_r18", "eval_r18"),
        "r50": prepare_eval_df(eval_r50, "pred_r50", "prob_r50", "correct_r50", "eval_r50"),
        "vit_tiny": prepare_eval_df(eval_vit_tiny, "pred_vit_tiny", "prob_vit_tiny", "correct_vit_tiny", "eval_vit_tiny"),
        "vit_base": prepare_eval_df(eval_vit_base, "pred_vit_base", "prob_vit_base", "correct_vit_base", "eval_vit_base"),
    }

    results_df = merge_model_results(candidate_df, eval_tables)

    print("[INFO] All candidates:", len(results_df))
    print("[INFO] All-correct candidates:", int(results_df["all_correct"].sum()))

    selected_df = (
        results_df[results_df["all_correct"] == 1]
        .sort_values(["parent_category", "wnid", "mean_prob"], ascending=[True, True, False])
        .groupby("wnid", group_keys=False)
        .head(args.max_images_per_class)
        .reset_index(drop=True)
    )

    print("[INFO] Selected images:", len(selected_df))

    save_selected_images(selected_df, output_dir)

    summary_per_parent = (
        selected_df.groupby("parent_category")
        .agg(
            num_selected_images=("file_name", "count"),
            num_wnids_with_selection=("wnid", "nunique"),
        )
        .sort_values("num_selected_images", ascending=False)
        .reset_index()
    )

    summary_per_wnid = (
        selected_df.groupby(["parent_category", "wnid"])
        .agg(num_selected_images=("file_name", "count"))
        .sort_values(["parent_category", "num_selected_images"], ascending=[True, False])
        .reset_index()
    )

    save_csv(candidate_df, output_dir / "candidate_images_prefiltered.csv")
    save_csv(results_df, output_dir / "all_model_results.csv")
    save_csv(selected_df, output_dir / "selected_images_all_categories.csv")
    save_csv(summary_per_parent, output_dir / "summary_per_parent_category.csv")
    save_csv(summary_per_wnid, output_dir / "summary_per_wnid.csv")


if __name__ == "__main__":
    main()
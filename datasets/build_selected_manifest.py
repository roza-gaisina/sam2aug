#!/usr/bin/env python3
"""
Build a ImageNet val manifest from a user-provided list of 256 flat filenames.

Input:
- a text file with one flat filename per line, e.g.:
    n04505470__ILSVRC2012_val_00000104.JPEG

Global lookup manifest from Imagenet (jsonl):
- entries like:
  {"image_id": "...", "wnid": "...", "rel_path": "...", "primary_bbox_xyxy": [...], ...}

Output:
- JSON list with entries:
  {
    "wnid": "...",
    "class_name": "...",              (optional mapping)
    "flat_filename": "...",
    "original_rel_path": "...",
    "primary_bbox_xyxy": [...],
    "image_id": "...",
    "image_size_hw": [H,W]
  }

Run using: 
Geirhos:
python build_selected_manifest.py \
  --selected-list /datasets/geirhos_selected_val_images.txt \
  --val-manifest /imagenet/meta/val_manifest.jsonl \
  --val-root /imagenet/val_classed \
  --out-json /datasets/inputs/imagenet/meta/val_selected_manifest.json \
  --class-map-json /datasets/geirhos_wnid_to_class_mapping.json

Object relocation experiments:
python build_selected_manifest.py \
  --selected-list /datasets/obj_reloc_selected_val_images.txt \
  --val-manifest /imagenet/meta/val_manifest.jsonl \
  --val-root /imagenet/val_classed \
  --out-json /datasets/inputs/imagenet/meta/val_obj_reloc_selected_manifest.json \
  --class-map-json /datasets/obj_reloc_wnid_to_class_mapping.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def read_lines(path: str) -> List[str]:
    lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def parse_flat_filename(flat: str) -> Tuple[str, str]:
    """
    flat: "n04505470__ILSVRC2012_val_00000104.JPEG" -> (wnid, filename)
    """
    if "__" not in flat:
        raise ValueError(f"Invalid flat filename (missing '__'): {flat}")
    wnid, fname = flat.split("__", 1)
    if not wnid.startswith("n") or len(wnid) != 9:
        raise ValueError(f"Flat filename has suspicious wnid '{wnid}' in: {flat}")
    if not fname:
        raise ValueError(f"Flat filename missing image filename part: {flat}")
    return wnid, fname


def image_id_from_filename(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def get_image_size_hw(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return (h, w)


def load_val_manifest_jsonl(path: str) -> Dict[Tuple[str, str], dict]:
    """
    Build a lookup: (wnid, filename) -> manifest entry
    using keys: wnid, rel_path, primary_bbox_xyxy, image_id (optional)
    """
    lookup = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            wnid = it.get("wnid")
            rel_path = it.get("rel_path") or it.get("original_rel_path")
            if wnid is None or rel_path is None:
                continue
            fname = os.path.basename(rel_path)
            lookup[(wnid, fname)] = it
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-list", required=True,
                    help="Text file with 256 flat filenames, one per line (wnid__ILSVRC2012_val_*.JPEG).")
    ap.add_argument("--val-manifest", required=True,
                    help="Global ImageNet val manifest jsonl (must contain wnid + rel_path + primary_bbox_xyxy).")
    ap.add_argument("--val-root", required=True,
                    help="Root of val_classed (contains <wnid>/ILSVRC2012_val_*.JPEG).")
    ap.add_argument("--out-json", required=True, help="Output manifest JSON path.")
    ap.add_argument("--class-map-json", default=None,
                    help="Optional JSON dict mapping wnid->class_name for nicer labels.")
    ap.add_argument("--expected-classes", type=int, default=11)
    #ap.add_argument("--expected-per-class", type=int, default=16)
    args = ap.parse_args()

    flats = read_lines(args.selected_list)

    #expected_total = args.expected_classes * args.expected_per_class
    expected_total = 55
    if len(flats) != expected_total:
        raise ValueError(f"Expected {expected_total} filenames, got {len(flats)} in {args.selected_list}")

    # duplicates check
    dupes = [k for k, v in Counter(flats).items() if v > 1]
    if dupes:
        raise ValueError(f"Found duplicate flat_filenames ({len(dupes)}). Example: {dupes[:5]}")

    # parse + count wnids
    wnids = []
    parsed = []
    for flat in flats:
        wnid, fname = parse_flat_filename(flat)
        wnids.append(wnid)
        parsed.append((flat, wnid, fname))

    wnid_counts = Counter(wnids)
    if len(wnid_counts) != args.expected_classes:
        raise ValueError(f"Expected {args.expected_classes} unique WNIDs, got {len(wnid_counts)}. "
                         f"Counts: {wnid_counts}")

    # bad_counts = {w: c for w, c in wnid_counts.items() if c != args.expected_per_class}
    # if bad_counts:
    #     raise ValueError(f"Expected exactly {args.expected_per_class} images per WNID. Bad counts: {bad_counts}")

    # optional class_name mapping
    wnid_to_name: Dict[str, str] = {}
    if args.class_map_json:
        class_map_path = os.path.expanduser(args.class_map_json)

        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"--class-map-json not found: {class_map_path}")

        with open(class_map_path, "r") as f:
            wnid_to_name = json.load(f)
        if not isinstance(wnid_to_name, dict):
            raise ValueError("--class-map-json must be a JSON dict of wnid->class_name")

    # load lookup
    lookup = load_val_manifest_jsonl(args.val_manifest)
    if not lookup:
        raise RuntimeError(f"Failed to build lookup from {args.val_manifest}. Is it jsonl and has wnid+rel_path?")

    out_entries = []
    missing = []

    for flat, wnid, fname in parsed:
        key = (wnid, fname)
        if key not in lookup:
            missing.append(flat)
            continue
        it = lookup[key]
        rel_path = it.get("rel_path") or it.get("original_rel_path")
        bbox = it.get("primary_bbox_xyxy")
        if rel_path is None or bbox is None:
            missing.append(flat)
            continue

        img_path = os.path.join(args.val_root, rel_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image referenced by manifest not found: {img_path}")

        h, w = get_image_size_hw(img_path)
        image_id = it.get("image_id") or image_id_from_filename(fname)

        out_entries.append({
            "wnid": wnid,
            "class_name": wnid_to_name.get(wnid, "UNKNOWN"),
            "flat_filename": flat,
            "original_rel_path": rel_path,
            "primary_bbox_xyxy": bbox,
            "image_id": image_id,
            "image_size_hw": [h, w],
        })

    if missing:
        raise KeyError(f"{len(missing)} selected filenames were not found in val manifest lookup. "
                       f"Examples: {missing[:10]}")

    # Final sanity: keep stable ordering (by wnid then flat_filename)
    out_entries.sort(key=lambda d: (d["wnid"], d["flat_filename"]))

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_entries, f, indent=2)

    print("OK ✅")
    print(f"Wrote {len(out_entries)} entries to: {args.out_json}")
    print("WNID counts:", dict(wnid_counts))


if __name__ == "__main__":
    main()
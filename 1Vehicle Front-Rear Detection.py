#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Identify preceding & following vehicles across multi-lane networks using FCD (floating car data).

Features
- CLI arguments (no hardcoded paths)
- Cross-platform paths (pathlib)
- Logging instead of prints/emojis
- Parallel processing with initializer to avoid repeated pickling
- Input validation with clear errors
- Progress bar (tqdm) optional
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict

# ----------------------------
# Globals for worker processes
# ----------------------------
FCD_DATA: pd.DataFrame | None = None
LANE_LENGTH_MAP: Dict[str, float] = {}
LANE_NEXT_MAP: Dict[str, List[str]] = {}
LANE_PREV_MAP: Dict[str, List[str]] = {}


def _init_worker(
    fcd_data: pd.DataFrame,
    lane_length_map: Dict[str, float],
    lane_next_map: Dict[str, List[str]],
    lane_prev_map: Dict[str, List[str]],
):
    """Initializer to set globals inside each worker process."""
    global FCD_DATA, LANE_LENGTH_MAP, LANE_NEXT_MAP, LANE_PREV_MAP
    FCD_DATA = fcd_data
    LANE_LENGTH_MAP = lane_length_map
    LANE_NEXT_MAP = lane_next_map
    LANE_PREV_MAP = lane_prev_map


# ----------------------------
# Core logic
# ----------------------------
def process_timestep_lane(data_tuple: Tuple[float, str, pd.DataFrame]) -> List[dict]:
    """
    Perform preceding/following vehicle matching for a single (timestep_time, lane_id) group.
    The worker relies on globals set by _init_worker.
    """
    timestep_time, lane_id, df = data_tuple
    lane_length = LANE_LENGTH_MAP.get(lane_id, 0.0)

    # ---------- Candidate preceding vehicles (next) ----------
    next_lane_list = LANE_NEXT_MAP.get(lane_id, [])
    next_lanes, next_offsets = [], []

    # Same lane: include in preceding-vehicle matching, offset = 0
    next_lanes.append(lane_id)
    next_offsets.append(0.0)

    # First layer: offset = length of the current lane
    for nl in next_lane_list:
        next_lanes.append(nl)
        next_offsets.append(lane_length)  # cumulative to first layer

    # Second layer: offset = current lane length + previous layer's next-lane length
    for nl in next_lane_list:
        nl_len = LANE_LENGTH_MAP.get(nl, 0.0)
        for n2 in LANE_NEXT_MAP.get(nl, []):
            next_lanes.append(n2)
            next_offsets.append(lane_length + nl_len)  # cumulative to second layer

    # Deduplicate: keep the smallest offset per lane
    tmp = {}
    for l, off in zip(next_lanes, next_offsets):
        tmp[l] = min(off, tmp.get(l, float("inf")))
    next_lanes = list(tmp.keys())
    next_offsets = [tmp[l] for l in next_lanes]
    next_length = dict(zip(next_lanes, next_offsets))

    # Filter candidates from global FCD at the same timestep
    next_df = (
        FCD_DATA[
            (FCD_DATA["vehicle_lane"].isin(next_lanes)) & (FCD_DATA["timestep_time"] == timestep_time)
        ]
        .copy()
        .reset_index(drop=True)
    )
    next_offset = next_df["vehicle_lane"].map(next_length).fillna(0.0)
    next_df["position_preceding"] = next_df["vehicle_pos"] + next_offset

    # ---------- Candidate following vehicles (prev) ----------
    previous_lane_list = LANE_PREV_MAP.get(lane_id, [])
    prev_lanes, prev_offsets = [], []

    # Same lane: include in following-vehicle matching, offset = 0
    prev_lanes.append(lane_id)
    prev_offsets.append(0.0)

    # First layer: offset = length of the predecessor lane
    for pl in previous_lane_list:
        prev_lanes.append(pl)
        pl_length = LANE_LENGTH_MAP.get(pl, 0.0)
        prev_offsets.append(pl_length)

    # Second layer: offset = len(first-layer predecessor) + len(second-layer predecessor)
    for pl in previous_lane_list:
        pl_len = LANE_LENGTH_MAP.get(pl, 0.0)
        for p2 in LANE_PREV_MAP.get(pl, []):
            prev_lanes.append(p2)
            pl2_len = LANE_LENGTH_MAP.get(p2, 0.0)
            prev_offsets.append(pl_len + pl2_len)

    prev_length = dict(zip(prev_lanes, prev_offsets))
    prev_df = (
        FCD_DATA[
            (FCD_DATA["vehicle_lane"].isin(prev_lanes)) & (FCD_DATA["timestep_time"] == timestep_time)
        ]
        .copy()
        .reset_index(drop=True)
    )
    prev_offset = prev_df["vehicle_lane"].map(prev_length).fillna(0.0)
    prev_df["position_following"] = prev_df["vehicle_pos"] - prev_offset

    # === Current vehicle data ===
    df = df.copy()
    df["position"] = df["vehicle_pos"]
    df = df.sort_values("position")
    next_df = next_df.sort_values("position_preceding").reset_index(drop=True)
    prev_df = prev_df.sort_values("position_following").reset_index(drop=True)

    # === Preceding vehicle matching (merge_asof) ===
    result_front = pd.merge_asof(
        df,
        next_df,
        left_on="position",
        right_on="position_preceding",
        direction="forward",
        allow_exact_matches=False,
        suffixes=("", "_preceding"),
    )
    result_front["following_headway_distance"] = result_front["position_preceding"] - result_front["position"]

    # === Following vehicle matching (merge_asof) ===
    result_both = pd.merge_asof(
        result_front,
        prev_df,
        left_on="position",
        right_on="position_following",
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_following"),
    )
    result_both["preceding_headway_distance"] = result_both["position"] - result_both["position_following"]

    # === Output fields ===
    result = result_both[
        [
            "vehicle_id",
            "timestep_time",
            "vehicle_id_preceding",
            "vehicle_pos_preceding",
            "vehicle_speed_preceding",
            "vehicle_lane_preceding",
            "following_headway_distance",
            "vehicle_id_following",
            "vehicle_pos_following",
            "vehicle_speed_following",
            "vehicle_lane_following",
            "preceding_headway_distance",
        ]
    ].rename(
        columns={
            "vehicle_id_preceding": "following_vehicle_id",
            "vehicle_pos_preceding": "following_flow_pos",
            "vehicle_speed_preceding": "following_vehicle_speed",
            "vehicle_lane_preceding": "following_vehicle_lane",
            "vehicle_id_following": "preceding_vehicle_id",
            "vehicle_pos_following": "preceding_flow_pos",
            "vehicle_speed_following": "preceding_vehicle_speed",
            "vehicle_lane_following": "preceding_vehicle_lane",
        }
    )

    return result.to_dict(orient="records")


# ----------------------------
# Utilities
# ----------------------------
def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def build_lane_maps(lane_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build one-to-many lane successor and predecessor maps."""
    next_map: Dict[str, List[str]] = defaultdict(list)
    prev_map: Dict[str, List[str]] = defaultdict(list)

    # Drop rows where lane_id or next_lane_id is NA
    w = lane_df.dropna(subset=["lane_id", "next_lane_id"])
    for _, row in w.iterrows():
        next_map[str(row["lane_id"])].append(str(row["next_lane_id"]))
        prev_map[str(row["next_lane_id"])].append(str(row["lane_id"]))

    return next_map, prev_map


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identify preceding & following vehicles from FCD using lane connectivity."
    )
    p.add_argument("--fcd", type=Path, required=True, help="Path to FCD csv file.")
    p.add_argument("--lane-attributes", type=Path, required=True, help="Path to lane attributes csv (lane_id, next_lane_id).")
    p.add_argument("--lane-lengths", type=Path, required=True, help="Path to lane lengths csv (lane_id|vehicle_lane, length).")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Directory to save results.")
    p.add_argument("--output-stem", type=str, default="neighbors_identification", help="Output file stem (without .csv).")
    p.add_argument("--max-workers", type=int, default=os.cpu_count() or 1, help="Max worker processes.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    logging.info("Loading inputs...")
    # Read lane data
    lane_df = pd.read_csv(args.lane_attributes)
    validate_columns(lane_df, ["lane_id", "next_lane_id"], "lane-attributes CSV")

    length_df = pd.read_csv(args.lane_lengths)
    # allow either 'lane_id' or 'vehicle_lane' in lengths file
    if "vehicle_lane" in length_df.columns:
        length_df = length_df.rename(columns={"vehicle_lane": "lane_id"})
    validate_columns(length_df, ["lane_id", "length"], "lane-lengths CSV")

    lane_length_map = dict(zip(length_df["lane_id"].astype(str), length_df["length"].astype(float)))
    lane_next_map, lane_prev_map = build_lane_maps(lane_df)

    # Read FCD
    fcd_df = pd.read_csv(args.fcd)
    validate_columns(
        fcd_df,
        ["timestep_time", "vehicle_id", "vehicle_lane", "vehicle_speed", "vehicle_pos"],
        "FCD CSV",
    )

    # Harmonize dtypes used in maps
    fcd_df["vehicle_lane"] = fcd_df["vehicle_lane"].astype(str)

    # Prepare global, minimal FCD view used by workers
    fcd_view = fcd_df[["timestep_time", "vehicle_id", "vehicle_lane", "vehicle_speed", "vehicle_pos"]].copy()

    # Build (timestep_time, lane_id) groups
    logging.info("Building (timestep_time, lane_id) task groups...")
    group_dict: Dict[Tuple[float, str], List[pd.DataFrame]] = defaultdict(list)
    for (timestep_time, lane_id), group in fcd_view.groupby(["timestep_time", "vehicle_lane"], sort=False):
        group_dict[(timestep_time, lane_id)].append(group)

    task_list = [(t, l, pd.concat(glist, ignore_index=True)) for (t, l), glist in group_dict.items()]
    logging.info("Number of grouped tasks: %d", len(task_list))

    # Run in parallel
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, int(args.max_workers))

    logging.info("Starting parallel processing with %d workers...", max_workers)
    all_results: List[dict] = []

    progress_iter = (
        tqdm(
            ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                                initargs=(fcd_view, lane_length_map, lane_next_map, lane_prev_map)
                                ).map(process_timestep_lane, task_list),
            total=len(task_list),
            disable=args.no_progress,
            desc="Processing",
        )
    )

    # NOTE: we keep the executor inside the tqdm iterator to ensure proper cleanup
    for res in progress_iter:
        all_results.extend(res)

    # Merge and save
    logging.info("Merging results...")
    results_df = pd.DataFrame(all_results)
    merged_df = pd.merge(fcd_df, results_df, on=["vehicle_id", "timestep_time"], how="left")
    merged_df.sort_values(by=["vehicle_id", "timestep_time"], inplace=True, ignore_index=True)

    save_path = args.output_dir / f"{args.output_stem}.csv"
    merged_df.to_csv(save_path, index=False, encoding="utf-8")

    elapsed_min = (time.time() - t0) / 60.0
    logging.info("Results saved to: %s", save_path)
    logging.info("Start time: %s", start_str)
    logging.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Total runtime: %.2f minutes", elapsed_min)


if __name__ == "__main__":
    main()

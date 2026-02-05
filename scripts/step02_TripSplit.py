# -*- coding: utf-8 -*-
"""
step02_TripSplit.py

Purpose
- Input: trajectory CSV after leader/follower matching (at least: vehicle_id, timestep_time, vehicle_speed)
- Output: trip-splitted CSV with added columns:
    trip_id, vehicle_acceleration, vehicle_jerk, distance_m, cumulative_distance, mileage_pos (optional)

Trip splitting logic (same as your original idea, but cleaner)
- Continuous zero-speed blocks are "stops"
- Non-zero blocks are "driving segments" => each driving segment is treated as one trip
- If exists, include the 1-second boundary points (t-1 and t+1) with speed==0 into the adjacent trip

Notes
- acceleration/jerk use forward difference with dt=1 (no division by dt), consistent with your original scripts
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
@dataclass
class TripSplitConfig:
    input_csv: str
    out_dir: str
    filename_prefix: str = "fcd"

    # Parallel processing (usually not necessary for this step)
    parallel: bool = False
    max_workers: int = 8

    # Use vehicle_pos as mileage origin (same as your original logic)
    pos_col: str = "vehicle_pos"

    # Core columns
    id_col: str = "vehicle_id"
    t_col: str = "timestep_time"
    v_col: str = "vehicle_speed"


def _split_trip_one_vehicle(g: pd.DataFrame, cfg: TripSplitConfig) -> pd.DataFrame:
    g = g.sort_values(cfg.t_col, kind="mergesort").reset_index(drop=True)

    v = g[cfg.v_col].to_numpy()
    t = g[cfg.t_col].to_numpy()

    is_zero = (v == 0)

    # block id for consecutive segments (zero/non-zero)
    block_id = (is_zero != np.r_[is_zero[0], is_zero[:-1]]).cumsum()
    g["_block"] = block_id

    # identify driving blocks (not all-zero)
    block_is_zero = g.groupby("_block")[cfg.v_col].apply(lambda s: (s == 0).all())
    drive_blocks = block_is_zero[~block_is_zero].index.tolist()

    # map each driving block to a segment index: 1..K
    drive_map = {b: i + 1 for i, b in enumerate(drive_blocks)}
    g["segment"] = g["_block"].map(drive_map).fillna(-1).astype(int)

    # include boundary t-1 / t+1 if speed==0
    if drive_blocks:
        seg_bounds = (
            g[g["segment"] > 0]
            .groupby("segment")[cfg.t_col]
            .agg(seg_t0="min", seg_t1="max")
            .reset_index()
        )

        t_set = set(t.tolist())
        for _, row in seg_bounds.iterrows():
            seg = int(row["segment"])
            t0 = row["seg_t0"]
            t1 = row["seg_t1"]

            t_prev = t0 - 1
            t_next = t1 + 1

            if t_prev in t_set:
                m = (g[cfg.t_col] == t_prev) & (g[cfg.v_col] == 0)
                g.loc[m, "segment"] = seg

            if t_next in t_set:
                m = (g[cfg.t_col] == t_next) & (g[cfg.v_col] == 0)
                g.loc[m, "segment"] = seg

    g.drop(columns=["_block"], inplace=True, errors="ignore")
    return g


def _add_kinematics(df: pd.DataFrame, cfg: TripSplitConfig) -> pd.DataFrame:
    df = df.sort_values([cfg.id_col, cfg.t_col], kind="mergesort").reset_index(drop=True)

    # forward diff, dt=1, no divide
    df["vehicle_acceleration"] = (
        df.groupby(cfg.id_col)[cfg.v_col].shift(-1) - df[cfg.v_col]
    ).fillna(0.0)

    df["vehicle_jerk"] = (
        df.groupby(cfg.id_col)["vehicle_acceleration"].shift(-1) - df["vehicle_acceleration"]
    ).fillna(0.0)

    # distance per step (same as your original formula)
    df["distance_m"] = df[cfg.v_col] + 0.5 * df["vehicle_acceleration"]

    # cumulative distance (shifted so the first point starts from 0)
    df["cumulative_distance"] = (
        df.groupby(cfg.id_col)["distance_m"].cumsum()
        .groupby(df[cfg.id_col]).shift(1, fill_value=0.0)
    )

    # optional mileage position
    if cfg.pos_col in df.columns:
        first_pos = df.groupby(cfg.id_col)[cfg.pos_col].first()
        df["mileage_pos"] = df[cfg.id_col].map(first_pos) + df["cumulative_distance"]
        df["mileage_pos"] = df["mileage_pos"].round(4)

    return df


def run(cfg: TripSplitConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.input_csv)

    # required columns check
    need = {cfg.id_col, cfg.t_col, cfg.v_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    # split by vehicle
    if cfg.parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        parts = []
        groups = [g.copy() for _, g in df.groupby(cfg.id_col, sort=False)]
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(_split_trip_one_vehicle, g, cfg) for g in groups]
            for fut in as_completed(futs):
                parts.append(fut.result())
        out = pd.concat(parts, ignore_index=True)
    else:
        out = df.groupby(cfg.id_col, group_keys=False, sort=False).apply(
            lambda g: _split_trip_one_vehicle(g, cfg)
        )

    # build trip_id from (vehicle_id, segment)
    out["trip_id"] = out.groupby([cfg.id_col, "segment"]).ngroup() + 1

    # cleanup temp column
    out.drop(columns=["segment"], inplace=True, errors="ignore")

    # add kinematics
    out = _add_kinematics(out, cfg)

    # final ordering & cleanup
    out = out.sort_values([cfg.id_col, cfg.t_col, "trip_id"], kind="mergesort").reset_index(drop=True)
    out.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    # output (EN filename)
    out_path = os.path.join(cfg.out_dir, f"{cfg.filename_prefix}_trip_split.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Trip splitting completed: {out_path}")
    return out_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="Split trips by zero-speed boundaries (dt=1 forward diff).")
    p.add_argument("--input_csv", required=True, help="Input trajectory CSV")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--prefix", default="fcd", help="Output filename prefix (default: fcd)")
    p.add_argument("--parallel", action="store_true", help="Parallel by vehicle_id")
    p.add_argument("--max_workers", type=int, default=8, help="Max workers for parallel mode")

    args = p.parse_args()

    cfg = TripSplitConfig(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        filename_prefix=args.prefix,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    run(cfg)


if __name__ == "__main__":
    main()

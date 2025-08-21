#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gurobi-based post-processing of vehicle trajectories with jerk/acceleration constraints.

Key features
------------
- CLI friendly (no hardcoded paths), cross-platform (pathlib).
- English comments/logging for GitHub users.
- Compatible with datasets using Chinese '工况ID' or English 'condition_id'.
- Parallel processing with chunked intermediate outputs (CSV or Parquet).
- Optional plotting of speed profiles per vehicle/condition.

Requirements
-----------
- Python 3.9+
- pandas, numpy, tqdm, matplotlib
- (Optional) pyarrow or fastparquet if you choose --chunk-format parquet
- Gurobi + gurobipy (valid license required)

Example
-------
python optimize_dynamics.py \
  --input data/sample_conditions.csv \
  --accel-envelope data/accel_envelope.csv \
  --jerk-envelope data/jerk_envelope.csv \
  --output-dir outputs \
  --tmp-dir outputs/tmp \
  --output-stem dynamics_constrained \
  --max-workers 8 \
  --chunk-format csv \
  --no-plots
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# matplotlib (plots are optional)
import matplotlib.pyplot as plt

# Try to import gurobipy with a clear error if missing
try:
    from gurobipy import Model, GRB, quicksum
except Exception as e:  # pragma: no cover
    raise ImportError(
        "gurobipy is required. Please install Gurobi and ensure a valid license is available.\n"
        "See: https://www.gurobi.com/documentation/ and pip install gurobipy"
    ) from e


# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
RNG = np.random.default_rng(SEED)


# ----------------------------
# CLI and logging
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize trajectories with jerk/acceleration constraints using Gurobi."
    )
    p.add_argument("--input", type=Path, required=True, help="Path to input CSV containing per-timestep data.")
    p.add_argument("--accel-envelope", type=Path, required=True,
                   help="CSV with acceleration envelope vs speed. "
                        "Expected columns: speed_kmh, q1_smooth, q99_smooth")
    p.add_argument("--jerk-envelope", type=Path, required=True,
                   help="CSV with jerk envelope vs acceleration grid. "
                        "Expected columns: accel_round, q1_smooth, q99_smooth")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Directory to save final CSV.")
    p.add_argument("--output-stem", type=str, default="dynamics_constrained", help="Output file stem (no extension).")
    p.add_argument("--tmp-dir", type=Path, default=Path("./outputs/tmp"),
                   help="Directory for intermediate chunk files.")
    p.add_argument("--chunk-format", choices=["csv", "parquet"], default="csv",
                   help="Intermediate file format. Parquet requires pyarrow/fastparquet.")
    p.add_argument("--batch-size", type=int, default=500, help="Number of results per intermediate chunk.")
    p.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1), help="Worker processes.")
    p.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit (seconds) per task.")
    p.add_argument("--threads", type=int, default=2, help="Gurobi threads per task.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    p.add_argument("--save-plots", type=Path, default=None,
                   help="Optional directory to save per-vehicle speed plots. If omitted, no plots are saved.")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Timestep in seconds used when integrating distance (default 1.0).")
    return p.parse_args()


def setup_logging(level: str):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ----------------------------
# Column utilities
# ----------------------------
def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def ensure_condition_column(df: pd.DataFrame) -> str:
    """
    Ensure there is a 'condition_id' column.
    Accepts Chinese '工况ID' and renames it to 'condition_id' for downstream code.
    Returns the unified column name: 'condition_id'.
    """
    if "condition_id" in df.columns:
        return "condition_id"
    if "工况ID" in df.columns:
        df.rename(columns={"工况ID": "condition_id"}, inplace=True)
        return "condition_id"
    raise ValueError("Input CSV must contain 'condition_id' or '工况ID' column.")


# ----------------------------
# Envelopes (acceleration & jerk)
# ----------------------------
def load_envelopes(accel_csv: Path, jerk_csv: Path):
    """
    Load envelope tables and return dictionaries:
      - Amin_dict, Amax_dict keyed by integer speed_kmh
      - accels list (sorted), JerkMin_dict, JerkMax_dict keyed by accel value
    """
    accel_env = pd.read_csv(accel_csv)
    validate_columns(accel_env, ["speed_kmh", "q1_smooth", "q99_smooth"], "accel-envelope CSV")
    speeds = sorted(accel_env["speed_kmh"].astype(int).unique())
    Amin_dict = dict(zip(speeds, accel_env["q1_smooth"].astype(float)))
    Amax_dict = dict(zip(speeds, accel_env["q99_smooth"].astype(float)))

    jerk_env = pd.read_csv(jerk_csv)
    validate_columns(jerk_env, ["accel_round", "q1_smooth", "q99_smooth"], "jerk-envelope CSV")
    accels = sorted(jerk_env["accel_round"].astype(float).unique())
    JerkMin_dict = dict(zip(accels, jerk_env["q1_smooth"].astype(float)))
    JerkMax_dict = dict(zip(accels, jerk_env["q99_smooth"].astype(float)))

    return Amin_dict, Amax_dict, accels, JerkMin_dict, JerkMax_dict


# ----------------------------
# Parameters & preparation
# ----------------------------
V_MAX = 20.0     # global speed cap (m/s)
A_UPPER = 5.0    # acceleration upper bound
A_LOWER = -4.0   # acceleration lower bound
JERK_UP = 4.0    # jerk upper bound
JERK_LOW = -5.0  # jerk lower bound


def build_prepared_df(df_all: pd.DataFrame, dt: float) -> pd.DataFrame:
    """
    Prepare derived columns: odometer, headway-based positions, etc.
    Assumes velocities are in m/s and timestep is 'dt' seconds.
    """
    df = df_all.copy()
    # Distance increment per timestep ~ v * dt  (original code used v with dt=1 implicitly)
    df["distance_m"] = df["vehicle_speed"].astype(float) * float(dt)
    df.sort_values(["vehicle_id", "timestep_time"], inplace=True)

    # Odometer: first position + cumulative distance
    first_pos = df.groupby("vehicle_id", sort=False)["vehicle_pos"].first()
    df["cumulative_distance"] = df.groupby("vehicle_id", sort=False)["distance_m"].cumsum()
    df["vehicle_odometer"] = df["vehicle_id"].map(first_pos) + df["cumulative_distance"]
    df["vehicle_odometer"] = df["vehicle_odometer"].round(4)

    # Headway safety (fallback to non-negative)
    min_fhd = float(df["following_headway_distance"].min())
    min_phd = float(df["preceding_headway_distance"].min())
    safe_distance = max(0.0, min(min_fhd, min_phd))

    # Positions of preceding/following vehicles in the same mileage frame
    df["preceding_mileage_pos"] = df["vehicle_odometer"] + df["preceding_headway_distance"]
    df["following_mileage_pos"] = df["vehicle_odometer"] - df["following_headway_distance"]

    return df, safe_distance


def make_speed_cap_map(df_all: pd.DataFrame) -> Dict[str, float]:
    """Per-vehicle speed cap based on observed max speed (m/s)."""
    cap = (
        df_all.groupby("vehicle_id", sort=False)["vehicle_speed"]
        .max()
        .astype(float)
        .to_dict()
    )
    return cap


def prepare_params_for_group(
    df_trip: pd.DataFrame,
    vehicle_id,
    safe_distance: float,
    accel_env_dicts: Tuple[Dict[int, float], Dict[int, float], List[float], Dict[float, float], Dict[float, float]],
    speed_cap_map: Dict[str, float],
) -> dict:
    """
    Build a parameter dictionary consumed by the Gurobi model for a single (vehicle, condition) group.
    """
    Amin_dict, Amax_dict, accels, JerkMin_dict, JerkMax_dict = accel_env_dicts

    # Time series (sorted)
    time_vals = sorted(df_trip["timestep_time"].unique())
    base_v = df_trip.set_index("timestep_time")["vehicle_speed"].to_dict()
    base_x = df_trip.set_index("timestep_time")["vehicle_odometer"].to_dict()

    # Preceding/following constraints active times
    preced_mask = df_trip["preceding_vehicle_id"].notna()
    follow_mask = df_trip["following_vehicle_id"].notna()
    preced_time = df_trip.loc[preced_mask, "timestep_time"].tolist()
    follow_time = df_trip.loc[follow_mask, "timestep_time"].tolist()
    preced_x = df_trip.set_index("timestep_time").loc[preced_time, "preceding_mileage_pos"].to_dict()
    follow_x = df_trip.set_index("timestep_time").loc[follow_time, "following_mileage_pos"].to_dict()

    # Vehicle-specific observed max speed
    v_max = float(speed_cap_map.get(vehicle_id, V_MAX))

    # Soft-constraint indicators (lock endpoints)
    b_v = {k: int(base_v[k] == 0) for k in base_v}
    if b_v:
        keys = list(b_v)
        b_v[keys[0]] = 1
        b_v[keys[-1]] = 1
    b_x = b_v.copy()

    params = {
        "vehicle_id": vehicle_id,
        "time": time_vals,
        "preced_time": set(preced_time),
        "follow_time": set(follow_time),
        "base_v": base_v,
        "base_x": base_x,
        "preced_x": preced_x,
        "follow_x": follow_x,
        "b_v": b_v,
        "b_x": b_x,
        "x_safe": safe_distance,
        "M": 1e6,
        "v_limit": V_MAX,
        "v_max": v_max,
        "a_upper": A_UPPER,
        "a_lower": A_LOWER,
        "jerk_upper": JERK_UP,
        "jerk_lower": JERK_LOW,
        "accel": accels,
        "Amin": Amin_dict,
        "Amax": Amax_dict,
        "JerkMin": JerkMin_dict,
        "JerkMax": JerkMax_dict,
        "w_a": 10.0,
        "w_v": 1.0,
        "w_x": 1.0,
    }
    return params


# ----------------------------
# Plotting
# ----------------------------
def plot_speed(car_data: pd.DataFrame, car_id: str, out_dir: Path):
    """Save speed comparison plot for a single vehicle/condition."""
    car_data = car_data.sort_values(by="timestep_time")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True, dpi=180)
    if "opt_v" in car_data.columns:
        ax.plot(car_data["timestep_time"], car_data["opt_v"] * 3.6, label="Post-processed (jerk-constrained)")
    if "base_v" in car_data.columns:
        ax.plot(car_data["timestep_time"], car_data["base_v"] * 3.6, label="Baseline (original)")

    ax.set_title("Vehicle Speed Over Time", fontsize=16)
    ax.set_xlabel("Timestep", fontsize=14)
    ax.set_ylabel("Speed (km/h)", fontsize=14)
    ax.set_ylim(0, max(90, (car_data.get("opt_v", car_data.get("base_v", pd.Series([0]))).max() or 0) * 4.2))
    ax.legend(title="Series", fontsize=12)
    ax.grid(True)

    safe_name = re.sub(r'[\\/*?:"<>|.]', "_", str(car_id))
    fig.savefig(out_dir / f"{safe_name}.png")
    plt.close(fig)


# ----------------------------
# Optimization model
# ----------------------------
def _align_piecewise_table(accels: List[float], table_dict: Dict) -> np.ndarray:
    """
    Align a dict (keyed by accel value or by 0..len(accels)-1 index) to the accel grid.
    Returns an array of length len(accels).
    """
    arr = np.zeros(len(accels), dtype=float)
    value_to_idx = {float(v): i for i, v in enumerate(accels)}

    try_as_value = True
    for k, v in table_dict.items():
        kf = float(k)
        if kf not in value_to_idx:
            try_as_value = False
            break
        arr[value_to_idx[kf]] = float(v)

    if not try_as_value:
        arr = np.zeros(len(accels), dtype=float)
        for k, v in table_dict.items():
            idx = int(k)
            if not (0 <= idx < len(accels)):
                raise ValueError(f"Piecewise-table key out of range: {idx}")
            arr[idx] = float(v)

    return arr


def run_gurobi(params: dict, cond_id, time_limit: int, threads: int, logger: logging.Logger) -> pd.DataFrame:
    """Build and solve the Gurobi model; return a result DataFrame for this (vehicle, condition)."""
    time_seq = list(params["time"])
    accels = list(params["accel"])
    if len(time_seq) < 2:
        # Not enough points to apply dynamic constraints
        return pd.DataFrame()

    times1 = time_seq[1:]
    prev_of = {time_seq[i]: time_seq[i - 1] for i in range(1, len(time_seq))}

    try:
        jmax_arr = _align_piecewise_table(accels, params["JerkMax"])
        jmin_arr = _align_piecewise_table(accels, params["JerkMin"])
    except Exception as e:
        logger.exception(f"[condition {cond_id}] Failed to align JerkMax/JerkMin: {e}")
        return pd.DataFrame()

    try:
        m = Model()
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", int(time_limit))
        m.setParam("Threads", int(threads))

        # Decision variables
        jerk = m.addVars(time_seq, lb=params["jerk_lower"], ub=params["jerk_upper"], name="jerk")
        a = m.addVars(time_seq, lb=params["a_lower"], ub=params["a_upper"], name="a")
        v = m.addVars(time_seq, lb=0.0, ub=params["v_limit"], name="v")
        x = m.addVars(time_seq, lb=-GRB.INFINITY, name="x")

        # SOS2 weights lambda_y(t) for t>t0, y in accel grid
        lam = {(t, y): m.addVar(lb=0.0, name=f"lam[{t},{y}]") for t in times1 for y in range(len(accels))}

        # SOS2 constraints & jerk bounds dependent on previous-step acceleration a[prev(t)]
        for t in times1:
            idxs = list(range(len(accels)))
            lam_vars = [lam[(t, y)] for y in idxs]

            # sum_y lam = 1
            m.addConstr(quicksum(lam_vars) == 1.0, name=f"lam_sum[{t}]")

            # Interpolate a[prev(t)] = sum_y accels[y] * lam_y
            tp = prev_of[t]
            m.addConstr(a[tp] == quicksum(accels[y] * lam[(t, y)] for y in idxs), name=f"a_interp[{t}]")

            # Declare SOS2 (ordered indices 0..Y-1)
            m.addSOS(GRB.SOS_TYPE2, lam_vars, idxs)

            # Interpolated jerk bounds at a[prev(t)]
            Jmax_at_a = quicksum(jmax_arr[y] * lam[(t, y)] for y in idxs)
            Jmin_at_a = quicksum(jmin_arr[y] * lam[(t, y)] for y in idxs)
            m.addConstr(jerk[tp] <= Jmax_at_a, name=f"j_up_from_a[{t}]")
            m.addConstr(jerk[tp] >= Jmin_at_a, name=f"j_lo_from_a[{t}]")

        # Kinematics (Δt = 1 in the original data; if your data uses another Δt, pre-scale inputs)
        for t in times1:
            tp = prev_of[t]
            m.addConstr(v[t] == v[tp] + a[tp] + 0.5 * jerk[tp], name=f"update_v[{t}]")
            m.addConstr(a[t] == a[tp] + jerk[tp], name=f"update_a[{t}]")
            m.addConstr(x[t] == x[tp] + (v[tp] + 0.5 * a[tp] + jerk[tp] / 6.0), name=f"update_x[{t}]")

        # Car-following safety constraints
        for t in params["preced_time"]:
            if t not in time_seq or t not in params["preced_x"]:
                continue
            m.addConstr(
                params["preced_x"][t] - x[t] >= params["x_safe"] + 0.1 * v[t],
                name=f"preced[{t}]",
            )
        for t in params["follow_time"]:
            if t not in time_seq or t not in params["follow_x"]:
                continue
            m.addConstr(
                x[t] - params["follow_x"][t] >= params["x_safe"] + 0.1 * params["v_max"],
                name=f"follow[{t}]",
            )

        # Soft constraints (big-M) around baseline path
        M_big = float(params["M"])
        bx = params["b_x"]
        bv = params["b_v"]
        for t in time_seq:
            m.addConstr(x[t] - params["base_x"][t] <= M_big * (1 - bx[t]), name=f"x_soft_u[{t}]")
            m.addConstr(x[t] - params["base_x"][t] >= -M_big * (1 - bx[t]), name=f"x_soft_l[{t}]")
            m.addConstr(v[t] - params["base_v"][t] <= M_big * (1 - bv[t]), name=f"v_soft_u[{t}]")
            m.addConstr(v[t] - params["base_v"][t] >= -M_big * (1 - bv[t]), name=f"v_soft_l[{t}]")
            m.addConstr(v[t] >= -M_big * bv[t], name=f"v_soft_l1[{t}]")  # relax > to >=

        # Objective: minimize acceleration energy + (optional) deviation from baseline speed
        obj = params["w_a"] * quicksum(a[t] * a[t] for t in time_seq)
        obj += params["w_v"] * quicksum(
            (v[t] - params["base_v"][t]) * (v[t] - params["base_v"][t]) * (1 - bv[t]) for t in time_seq
        )
        # If needed, add position anchoring term similarly with (1 - bx[t]) weighting.

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        status = m.Status
        solcnt = m.SolCount
        logger.info(f"[condition {cond_id}] Gurobi status: {status}, solutions: {solcnt}")

        if solcnt > 0:
            df_out = pd.DataFrame(
                {
                    "timestep_time": time_seq,
                    "vehicle_id": params["vehicle_id"],
                    "maxSpeed": params["v_max"],
                    "opt_a": [a[t].X for t in time_seq],
                    "opt_v": [v[t].X for t in time_seq],
                    "opt_x": [x[t].X for t in time_seq],
                    "opt_jerk": [jerk[t].X for t in time_seq],
                    "base_v": [params["base_v"][t] for t in time_seq],
                }
            )
            return df_out

        # Infeasible or no solution found
        return pd.DataFrame(
            {
                "timestep_time": time_seq,
                "vehicle_id": params["vehicle_id"],
                "maxSpeed": params["v_max"],
                "opt_a": [np.nan] * len(time_seq),
                "opt_v": [np.nan] * len(time_seq),
                "opt_x": [np.nan] * len(time_seq),
                "opt_jerk": [np.nan] * len(time_seq),
                "base_v": [params["base_v"][t] for t in time_seq],
            }
        )

    except Exception as e:
        logger.exception(f"[condition {cond_id}] Gurobi exception: {e}")
        return pd.DataFrame()


# ----------------------------
# Task wrapper (for executors)
# ----------------------------
def process_task(task):
    (veh_id, cond_id, df_grp, safe_distance, env_dicts, speed_cap_map, time_limit, threads) = task
    logger = logging.getLogger("worker")
    params = prepare_params_for_group(df_grp, veh_id, safe_distance, env_dicts, speed_cap_map)
    return run_gurobi(params, cond_id, time_limit=time_limit, threads=threads, logger=logger)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("main")

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading inputs...")
    df_all = pd.read_csv(args.input)
    cond_col = ensure_condition_column(df_all)

    required_cols = [
        "timestep_time",
        "vehicle_id",
        "vehicle_speed",
        "vehicle_pos",
        "following_headway_distance",
        "preceding_headway_distance",
    ]
    validate_columns(df_all, required_cols, "input CSV")

    # Build derived fields (odometer, headway positions) and safety distance
    df_all, safe_distance = build_prepared_df(df_all, dt=args.dt)
    log.info("Computed safe distance (min headway): %.3f", safe_distance)

    # Speed cap per vehicle (based on observed max speed)
    speed_cap_map = make_speed_cap_map(df_all)

    # Load envelopes
    env_dicts = load_envelopes(args.accel_envelope, args.jerk_envelope)

    # Remove conditions with all-zero speeds (optional cleanup similar to original)
    df_clean = df_all.groupby(cond_col, sort=False).filter(lambda g: (g["vehicle_speed"] != 0).any())
    df_clean = df_clean.sort_values([cond_col, "vehicle_id"], ascending=[False, False])

    # Build tasks per (vehicle_id, condition_id)
    tasks = [
        (veh_id, cond_id, df_grp, safe_distance, env_dicts, speed_cap_map, args.time_limit, args.threads)
        for (veh_id, cond_id), df_grp in df_clean.groupby(["vehicle_id", cond_col], sort=False)
    ]
    log.info("Total tasks: %d", len(tasks))

    # Parallel solve with chunked intermediate outputs
    batch = max(1, int(args.batch_size))
    buf: List[pd.DataFrame] = []
    part_id = 0

    progress_iter = tqdm(
        ProcessPoolExecutor(max_workers=int(args.max_workers)).map(process_task, tasks, chunksize=8),
        total=len(tasks),
        desc="Condition optimization",
        disable=bool(args.no_progress),
    )

    for i, df_res in enumerate(progress_iter):
        if df_res is not None and not df_res.empty:
            buf.append(df_res)

        if (i + 1) % batch == 0 and buf:
            part_path = args.tmp_dir / f"part_{part_id:05d}.{args.chunk_format}"
            if args.chunk_format == "parquet":
                df_cat = pd.concat(buf, ignore_index=True)
                df_cat.to_parquet(part_path)
            else:
                df_cat = pd.concat(buf, ignore_index=True)
                df_cat.to_csv(part_path, index=False)
            part_id += 1
            buf.clear()
            gc.collect()

    # Flush remainder
    if buf:
        part_path = args.tmp_dir / f"part_{part_id:05d}.{args.chunk_format}"
        if args.chunk_format == "parquet":
            pd.concat(buf, ignore_index=True).to_parquet(part_path)
        else:
            pd.concat(buf, ignore_index=True).to_csv(part_path, index=False)
        buf.clear()
        gc.collect()

    # Gather all parts
    parts = sorted(p for p in args.tmp_dir.glob(f"*.{args.chunk_format}"))
    if not parts:
        log.warning("No intermediate parts found; nothing to merge.")
        return

    log.info("Merging %d parts...", len(parts))
    if args.chunk_format == "parquet":
        df_results = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df_results = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    # Merge back to original rows
    df_results = df_results.sort_values(["vehicle_id", "timestep_time"])
    df_final = pd.merge(df_all, df_results, how="left", on=["vehicle_id", "timestep_time"])

    # Select common/useful columns (only keep those that exist)
    preferred_cols = [
        "vehicle_id",
        "timestep_time",
        cond_col,
        "vehicle_type",
        "vehicle_odometer",
        "vehicle_speed",
        "vehicle_x",
        "vehicle_y",
        "opt_jerk",
        "opt_a",
        "opt_v",
        "opt_x",
    ]
    use_cols = [c for c in preferred_cols if c in df_final.columns]
    df_out = df_final[use_cols]

    out_path = args.output_dir / f"{args.output_stem}.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    log.info("Saved final output to: %s", out_path)

    # Optional plotting per (vehicle, condition)
    if args.save_plots is not None:
        plot_dir = Path(args.save_plots)
        log.info("Saving plots to: %s", plot_dir)
        for (veh_id, cond_id), grp in tqdm(
            df_final.groupby(["vehicle_id", cond_col], sort=False),
            total=df_final.groupby(["vehicle_id", cond_col], sort=False).ngroups,
            desc="Plotting",
            disable=bool(args.no_progress),
        ):
            # Compose ID for filename
            plot_id = f"{veh_id}_{cond_id}"
            plot_speed(grp[["timestep_time", "opt_v", "base_v"]], plot_id, plot_dir)

    elapsed_min = (time.time() - t0) / 60.0
    log.info("Total runtime: %.2f minutes", elapsed_min)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
step03_merge_fill_xy.py

Merge 2~4 trajectory CSVs (SUMO + processed methods) by vertical concatenation, fill missing
vehicle_accel / vehicle_jerk / vehicle_odometer when absent, fill vehicle_type from SUMO,
and fill vehicle_x / vehicle_y for non-SUMO methods via odometer-based interpolation using SUMO as reference.

Optional plotting (per-vehicle):
- XY tracks comparison across data_type (vehicle_x vs vehicle_y)
- Speed/Accel/Jerk comparison across data_type (3-panel)

Plotting is disabled by default. When enabled, only a fixed number of vehicle_ids will be plotted,
selected by a fixed random seed for reproducibility.

Requirements:
- SUMO CSV must contain:
  timestep_time, vehicle_id, trip_id, vehicle_speed, vehicle_x, vehicle_y, vehicle_type
  (vehicle_accel / vehicle_jerk / vehicle_odometer are optional; computed if missing)

- Non-SUMO CSV(s) must contain at least:
  timestep_time, vehicle_id, trip_id, vehicle_speed
  Optional: vehicle_accel, vehicle_jerk, vehicle_odometer (computed if missing)

Output columns (minimum):
  timestep_time, vehicle_id, vehicle_speed, vehicle_accel, vehicle_odometer,
  vehicle_type, vehicle_x, vehicle_y, data_type

Notes:
- All messages, labels, and comments are English only.
- No interactive prompts; use CLI arguments.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ------------------------------ Constants ------------------------------

DEFAULT_RANDOM_SEED = 42
DEFAULT_PLOT_N = 10


# ------------------------------ Logging ------------------------------

def setup_logger(log_path: str | None) -> logging.Logger:
    logger = logging.getLogger("step03_merge_fill_xy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ------------------------------ Utilities ------------------------------

REQUIRED_MIN_NON_SUMO = {"timestep_time", "vehicle_id", "trip_id", "vehicle_speed"}
REQUIRED_MIN_SUMO = REQUIRED_MIN_NON_SUMO | {"vehicle_x", "vehicle_y", "vehicle_type"}


def _clean_path(s: str) -> str:
    return s.strip().strip('"').strip("'")


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|.\s]+', "_", str(name)).strip("_")


def _compute_accel_jerk_odometer(
    df: pd.DataFrame,
    dt: float,
    need_accel: bool,
    need_jerk: bool,
    need_odo: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["timestep_time"] = _safe_numeric(out["timestep_time"])
    out["vehicle_speed"] = _safe_numeric(out["vehicle_speed"])

    out.sort_values(["vehicle_id", "timestep_time"], inplace=True, kind="mergesort")

    if need_accel:
        out["vehicle_accel"] = (
            out.groupby("vehicle_id", sort=False)["vehicle_speed"].diff().fillna(0.0) / float(dt)
        )

    if need_jerk:
        if "vehicle_accel" not in out.columns:
            out["vehicle_accel"] = (
                out.groupby("vehicle_id", sort=False)["vehicle_speed"].diff().fillna(0.0) / float(dt)
            )
        out["vehicle_jerk"] = (
            out.groupby("vehicle_id", sort=False)["vehicle_accel"].diff().fillna(0.0) / float(dt)
        )

    if need_odo:
        if "vehicle_accel" not in out.columns:
            out["vehicle_accel"] = (
                    out.groupby("vehicle_id", sort=False)["vehicle_speed"].diff().fillna(0.0) / float(dt)
            )

        delta_s = out["vehicle_speed"].fillna(0.0) * float(dt) + 0.5 * out["vehicle_accel"].fillna(0.0) * (
                    float(dt) ** 2)

        out["vehicle_odometer"] = (
            delta_s.groupby(out["vehicle_id"], sort=False).cumsum()
        )

    return out


def read_and_standardize(
    csv_path: str,
    data_type: str,
    is_sumo: bool,
    dt: float,
) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    df = pd.read_csv(p)

    # Basic compatibility for legacy column names
    rename_map = {
        "car_speed": "vehicle_speed",
        "car_accel": "vehicle_accel",
        "car_jerk": "vehicle_jerk",
        "coord_x": "vehicle_x",
        "coord_y": "vehicle_y",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if is_sumo:
        _ensure_cols(df, list(REQUIRED_MIN_SUMO))
    else:
        _ensure_cols(df, list(REQUIRED_MIN_NON_SUMO))

    # Create missing optional columns
    for c in ["vehicle_accel", "vehicle_jerk", "vehicle_odometer", "vehicle_type", "vehicle_x", "vehicle_y"]:
        if c not in df.columns:
            df[c] = np.nan

    # Compute missing accel/jerk/odometer only if all-NaN
    need_accel = df["vehicle_accel"].isna().all()
    need_jerk = df["vehicle_jerk"].isna().all()
    need_odo = df["vehicle_odometer"].isna().all()

    df = _compute_accel_jerk_odometer(df, dt, need_accel, need_jerk, need_odo)

    # Add data_type
    df["data_type"] = str(data_type)

    # Cast identifiers
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    df["trip_id"] = df["trip_id"].astype(str)

    keep_cols = [
        "timestep_time",
        "vehicle_id",
        "trip_id",
        "vehicle_speed",
        "vehicle_accel",
        "vehicle_jerk",
        "vehicle_odometer",
        "vehicle_type",
        "vehicle_x",
        "vehicle_y",
        "data_type",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df.sort_values(["vehicle_id", "timestep_time"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df


def build_vehicle_type_map(df_sumo: pd.DataFrame) -> Dict[str, str]:
    tmp = df_sumo[["vehicle_id", "vehicle_type"]].copy()
    tmp["vehicle_id"] = tmp["vehicle_id"].astype(str)
    tmp["vehicle_type"] = tmp["vehicle_type"].astype(str)
    tmp = tmp[tmp["vehicle_type"].notna() & (tmp["vehicle_type"].astype(str) != "nan")]
    tmp = tmp.drop_duplicates(subset=["vehicle_id"], keep="first")
    return dict(zip(tmp["vehicle_id"], tmp["vehicle_type"]))


def fill_vehicle_type(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    if "vehicle_type" not in out.columns:
        out["vehicle_type"] = np.nan
    out["vehicle_type"] = out["vehicle_type"].where(out["vehicle_type"].notna(), out["vehicle_id"].map(type_map))
    return out


# ------------------------------ XY interpolation ------------------------------

def _interp_xy_for_one_vehicle(df_vehicle: pd.DataFrame) -> pd.DataFrame:
    vdf = df_vehicle.copy()

    known = vdf[
        (vdf["data_type"] == "SUMO")
        & vdf["vehicle_odometer"].notna()
        & vdf["vehicle_x"].notna()
        & vdf["vehicle_y"].notna()
    ][["vehicle_odometer", "vehicle_x", "vehicle_y"]].copy()

    if len(known) < 2:
        return vdf

    known["vehicle_odometer"] = pd.to_numeric(known["vehicle_odometer"], errors="coerce")
    known["vehicle_x"] = pd.to_numeric(known["vehicle_x"], errors="coerce")
    known["vehicle_y"] = pd.to_numeric(known["vehicle_y"], errors="coerce")
    known = known.dropna()

    if len(known) < 2:
        return vdf

    known.sort_values("vehicle_odometer", inplace=True, kind="mergesort")
    known = known.drop_duplicates(subset=["vehicle_odometer"], keep="first")
    if len(known) < 2:
        return vdf

    odo_known = known["vehicle_odometer"].to_numpy(dtype=float)
    x_known = known["vehicle_x"].to_numpy(dtype=float)
    y_known = known["vehicle_y"].to_numpy(dtype=float)

    fx = interp1d(odo_known, x_known, kind="linear", fill_value="extrapolate", bounds_error=False)
    fy = interp1d(odo_known, y_known, kind="linear", fill_value="extrapolate", bounds_error=False)

    valid = vdf["vehicle_odometer"].notna()
    if not valid.any():
        return vdf

    odo_all = pd.to_numeric(vdf.loc[valid, "vehicle_odometer"], errors="coerce").to_numpy(dtype=float)
    vdf.loc[valid, "vehicle_x"] = fx(odo_all)
    vdf.loc[valid, "vehicle_y"] = fy(odo_all)
    return vdf


def interpolate_xy_parallel(df_all: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    groups = [g for _, g in df_all.groupby("vehicle_id", sort=False)]

    if max_workers <= 1:
        out_list = []
        for g in tqdm(groups, total=len(groups), desc="Interpolating XY"):
            out_list.append(_interp_xy_for_one_vehicle(g))
        return pd.concat(out_list, ignore_index=True)

    out_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(ex.map(_interp_xy_for_one_vehicle, groups), total=len(groups), desc="Interpolating XY"):
            out_list.append(res)

    return pd.concat(out_list, ignore_index=True)


# ------------------------------ Plot selection ------------------------------

def select_vehicle_ids_for_plot(
    df_all: pd.DataFrame,
    plot_n: int,
    random_seed: int,
    logger: logging.Logger,
) -> List[str]:
    vehicle_ids = sorted(pd.unique(df_all["vehicle_id"].astype(str)))
    total = len(vehicle_ids)

    logger.info(f"Random seed = {random_seed}")
    logger.info(f"Total vehicles = {total}")

    if total == 0:
        logger.info("No vehicles found. Plotting will be skipped.")
        return []

    if plot_n is None or int(plot_n) <= 0:
        logger.info(f"Invalid plot_n={plot_n}. Fallback to default={DEFAULT_PLOT_N}.")
        plot_n = DEFAULT_PLOT_N
    plot_n = int(plot_n)

    if plot_n >= total:
        logger.info(f"Requested plot_n = {plot_n} >= total. Plotting all vehicles.")
        return vehicle_ids

    rng = np.random.default_rng(int(random_seed))
    chosen = rng.choice(vehicle_ids, size=plot_n, replace=False).tolist()
    chosen = sorted(chosen)
    logger.info(f"Requested plot_n = {plot_n}. Selected vehicles = {len(chosen)}.")
    return chosen


# ------------------------------ Plotting: XY tracks ------------------------------

def _plot_xy_one_vehicle(payload: Tuple[str, pd.DataFrame, str]) -> None:
    vehicle_id, car_df, out_dir = payload
    car_df = car_df.sort_values("timestep_time")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200, constrained_layout=True)

    for dt_name, g in car_df.groupby("data_type", sort=False):
        ax.plot(g["vehicle_x"], g["vehicle_y"], label=str(dt_name), linewidth=1.2, alpha=0.85)

    ax.set_title(f"vehicle_id = {vehicle_id}")
    ax.set_xlabel("vehicle_x")
    ax.set_ylabel("vehicle_y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fp = os.path.join(out_dir, f"{safe_filename(vehicle_id)}.png")
    plt.savefig(fp)
    plt.close(fig)


def plot_xy_tracks_subset(
    df_all: pd.DataFrame,
    selected_vehicle_ids: List[str],
    out_dir: str,
    max_workers: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    sel = set(str(x) for x in selected_vehicle_ids)
    tasks = [(vid, g.copy(), out_dir) for vid, g in df_all.groupby("vehicle_id", sort=False) if str(vid) in sel]

    if not tasks:
        return

    if max_workers <= 1:
        for t in tqdm(tasks, total=len(tasks), desc="Plotting XY"):
            _plot_xy_one_vehicle(t)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_plot_xy_one_vehicle, tasks), total=len(tasks), desc="Plotting XY"))


# ------------------------------ Plotting: Speed/Accel/Jerk ------------------------------

def _ensure_saj_columns_for_plot(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    out = df.copy()
    out["timestep_time"] = pd.to_numeric(out["timestep_time"], errors="coerce")
    out["vehicle_speed"] = pd.to_numeric(out["vehicle_speed"], errors="coerce")

    # Compute accel if missing or all-NaN
    if ("vehicle_accel" not in out.columns) or out["vehicle_accel"].isna().all():
        out["vehicle_accel"] = (
            out.groupby(["data_type", "vehicle_id"], sort=False)["vehicle_speed"].diff().fillna(0.0) / float(dt)
        )
    else:
        out["vehicle_accel"] = pd.to_numeric(out["vehicle_accel"], errors="coerce").fillna(0.0)

    # Compute jerk if missing or all-NaN
    if ("vehicle_jerk" not in out.columns) or out["vehicle_jerk"].isna().all():
        out["vehicle_jerk"] = (
            out.groupby(["data_type", "vehicle_id"], sort=False)["vehicle_accel"].diff().fillna(0.0) / float(dt)
        )
    else:
        out["vehicle_jerk"] = pd.to_numeric(out["vehicle_jerk"], errors="coerce").fillna(0.0)

    return out


def _plot_saj_one_vehicle(payload: Tuple[str, pd.DataFrame, str, float, str]) -> None:
    vehicle_id, car_df, out_dir, dt, speed_unit = payload
    if car_df.empty:
        return

    df = car_df.copy()
    df["timestep_time"] = pd.to_numeric(df["timestep_time"], errors="coerce")
    df = df[df["timestep_time"].notna()].copy()
    if df.empty:
        return

    df.sort_values(["data_type", "timestep_time"], inplace=True, kind="mergesort")
    df = _ensure_saj_columns_for_plot(df, dt=dt)

    # Speed unit
    if speed_unit.lower() in ["km/h", "kmh", "kph"]:
        df["vehicle_speed_plot"] = df["vehicle_speed"] * 3.6
        y_speed_label = "Speed (km/h)"
    else:
        df["vehicle_speed_plot"] = df["vehicle_speed"]
        y_speed_label = "Speed (m/s)"

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(11, 9),
        sharex=True, constrained_layout=True, dpi=150
    )
    ax_speed, ax_accel, ax_jerk = axes

    for dt_name, g in df.groupby("data_type", sort=False):
        if g.empty:
            continue
        label = str(dt_name)
        ax_speed.plot(g["timestep_time"], g["vehicle_speed_plot"], label=label, linewidth=1.5, alpha=0.85)
        ax_accel.plot(g["timestep_time"], g["vehicle_accel"], label=label, linewidth=1.5, alpha=0.85)
        ax_jerk.plot(g["timestep_time"], g["vehicle_jerk"], label=label, linewidth=1.5, alpha=0.85)

    ax_speed.set_title(f"vehicle_id = {vehicle_id}", fontweight="bold")
    ax_speed.set_ylabel(y_speed_label, fontweight="bold")
    ax_accel.set_ylabel(r"Acceleration (m/s$^2$)", fontweight="bold")
    ax_jerk.set_ylabel(r"Jerk (m/s$^3$)", fontweight="bold")
    ax_jerk.set_xlabel("Simulation time (s)", fontweight="bold")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, color="0.8")

    ax_speed.legend(loc="best", ncol=1)

    fp = os.path.join(out_dir, f"{safe_filename(vehicle_id)}.png")
    plt.savefig(fp)
    plt.close(fig)


def plot_saj_subset(
    df_all: pd.DataFrame,
    selected_vehicle_ids: List[str],
    out_dir: str,
    max_workers: int,
    dt: float,
    speed_unit: str = "m/s",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    sel = set(str(x) for x in selected_vehicle_ids)
    tasks = [(vid, g.copy(), out_dir, dt, speed_unit) for vid, g in df_all.groupby("vehicle_id", sort=False) if str(vid) in sel]

    if not tasks:
        return

    if max_workers <= 1:
        for t in tqdm(tasks, total=len(tasks), desc="Plotting SAJ"):
            _plot_saj_one_vehicle(t)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_plot_saj_one_vehicle, tasks), total=len(tasks), desc="Plotting SAJ"))


# ------------------------------ Main ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step03: merge CSVs, fill XY by SUMO interpolation, optional per-vehicle plots.")
    ap.add_argument("--sumo_csv", required=True, help="Path to SUMO CSV (must contain vehicle_x, vehicle_y, vehicle_type).")
    ap.add_argument("--rvst_csv", required=True, help="Path to RVST CSV (min: timestep_time, vehicle_id, trip_id, vehicle_speed).")

    ap.add_argument("--use_sg_smooth", action="store_true", help="Enable reading SG CSV.")
    ap.add_argument("--sg_csv", default=None, help="Path to SG CSV (used only when --use_sg_smooth is set).")

    ap.add_argument("--m4_csv", default=None, help="Path to method4 CSV (optional).")
    ap.add_argument("--m4_type", default="METHOD4", help='data_type label for method4 (default: "METHOD4").')

    ap.add_argument("--dt", type=float, default=1.0, help="Time step size in seconds (default: 1.0).")
    ap.add_argument("--max_workers", type=int, default=0, help="Process workers (0=auto, 1=serial).")

    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--keep_trip_and_jerk", action="store_true", help="Keep trip_id and vehicle_jerk in output CSV.")

    # Plot controls (default: off)
    ap.add_argument("--plot_xy", action="store_true", help="Plot XY tracks for selected vehicles (default: off).")
    ap.add_argument("--plot_saj", action="store_true", help="Plot speed/accel/jerk for selected vehicles (default: off).")
    ap.add_argument("--plot_n", type=int, default=DEFAULT_PLOT_N, help=f"Number of vehicles to plot (default: {DEFAULT_PLOT_N}).")
    ap.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"Random seed for selecting vehicles (default: {DEFAULT_RANDOM_SEED}).")
    ap.add_argument("--plot_dir", default=None, help="Directory for plots (default: alongside output CSV).")
    ap.add_argument("--saj_speed_unit", default="m/s", help='Speed unit in SAJ plot: "m/s" or "km/h" (default: m/s).')

    ap.add_argument("--log_path", default=None, help="Optional log file path.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_path)

    dt = float(args.dt)
    if dt <= 0:
        raise ValueError("--dt must be > 0")

    # Determine workers
    if args.max_workers and args.max_workers > 0:
        max_workers = int(args.max_workers)
    else:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - 1)

    logger.info("Reading inputs...")
    df_sumo = read_and_standardize(args.sumo_csv, data_type="SUMO", is_sumo=True, dt=dt)
    df_rvst = read_and_standardize(args.rvst_csv, data_type="RVST", is_sumo=False, dt=dt)

    dfs = [df_sumo, df_rvst]

    if args.use_sg_smooth:
        if not args.sg_csv:
            raise ValueError("--sg_csv must be provided when --use_sg_smooth is set")
        df_sg = read_and_standardize(args.sg_csv, data_type="SG-BD", is_sumo=False, dt=dt)
        dfs.append(df_sg)

    if args.m4_csv:
        df_m4 = read_and_standardize(args.m4_csv, data_type=str(args.m4_type), is_sumo=False, dt=dt)
        dfs.append(df_m4)

    logger.info("Filling vehicle_type for non-SUMO methods using SUMO mapping...")
    type_map = build_vehicle_type_map(df_sumo)
    dfs_filled = [dfs[0]] + [fill_vehicle_type(d, type_map) for d in dfs[1:]]

    logger.info("Concatenating datasets...")
    df_all = pd.concat(dfs_filled, ignore_index=True)

    df_all.sort_values(["data_type", "vehicle_id", "timestep_time"], inplace=True, kind="mergesort")
    df_all.drop_duplicates(subset=["data_type", "vehicle_id", "timestep_time"], keep="first", inplace=True, ignore_index=True)

    logger.info("Interpolating vehicle_x/vehicle_y for all methods using SUMO reference...")
    df_all = interpolate_xy_parallel(df_all, max_workers=max_workers)

    df_all.sort_values(["data_type", "vehicle_id", "timestep_time"], inplace=True, kind="mergesort")
    df_all.reset_index(drop=True, inplace=True)

    # Output columns
    base_cols = [
        "timestep_time",
        "vehicle_id",
        "vehicle_speed",
        "vehicle_accel",
        "vehicle_odometer",
        "vehicle_type",
        "vehicle_x",
        "vehicle_y",
        "data_type",
    ]
    extra_cols = ["trip_id", "vehicle_jerk"] if args.keep_trip_and_jerk else []
    out_cols = base_cols + [c for c in extra_cols if c in df_all.columns]

    for c in base_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan

    out_path = Path(_clean_path(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Saved merged CSV: {out_path}")

    # Optional plotting
    if args.plot_xy or args.plot_saj:
        plot_dir = args.plot_dir or str(out_path.parent / "plots")
        os.makedirs(plot_dir, exist_ok=True)

        selected_ids = select_vehicle_ids_for_plot(
            df_all=df_all,
            plot_n=int(args.plot_n),
            random_seed=int(args.random_seed),
            logger=logger,
        )

        if args.plot_xy:
            logger.info("Plotting XY tracks...")
            plot_xy_tracks_subset(
                df_all=df_all,
                selected_vehicle_ids=selected_ids,
                out_dir=os.path.join(plot_dir, "xy_tracks"),
                max_workers=max_workers,
            )

        if args.plot_saj:
            logger.info("Plotting speed/accel/jerk...")
            plot_saj_subset(
                df_all=df_all,
                selected_vehicle_ids=selected_ids,
                out_dir=os.path.join(plot_dir, "speed_accel_jerk"),
                max_workers=max_workers,
                dt=dt,
                speed_unit=str(args.saj_speed_unit),
            )

        logger.info(f"Plots saved under: {plot_dir}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

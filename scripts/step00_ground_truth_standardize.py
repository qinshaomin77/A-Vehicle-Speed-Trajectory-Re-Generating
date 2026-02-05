# -*- coding: utf-8 -*-
"""
step00_ground_truth_standardize.py

Purpose
-------
Optional Ground-Truth (GT) CSV standardizer for GUI pipelines.

- If a ground-truth CSV is provided, this script:
  1) Unifies column names to the project standard:
     'timestep_time', 'vehicle_id', 'vehicle_speed', 'vehicle_accel',
     'vehicle_jerk', 'vehicle_odometer', 'vehicle_type', 'vehicle_x', 'vehicle_y'
  2) Assumes vehicle_speed unit is m/s by default (can override via CLI).
  3) Enforces speed-acceleration integral/differential consistency:
     - If accel missing: compute from speed (diff) using per-vehicle dt.
     - If speed missing: reconstruct from accel (integral) with an initial speed (default 0).
     - If both exist and --prefer speed: recompute accel from speed (default).
       If both exist and --prefer accel: recompute speed from accel.
  4) Computes odometer by integrating speed over time if missing.
  5) Does NOT compute jerk by default; outputs 'vehicle_jerk' as NaN (placeholder)
     unless the input already contains jerk and --keep_input_jerk is enabled.

- If no GT file is provided in your GUI pipeline, simply skip calling this script.

Usage
-----
python step00_ground_truth_standardize.py --in_csv path/to/ground_truth.csv --out_dir path/to/output

Outputs
-------
{out_dir}/00_ground_truth/ground_truth_std.csv
{out_dir}/00_ground_truth/ground_truth_std_report.json
{out_dir}/00_ground_truth/ground_truth_std.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Config & report -----------------------------

STANDARD_COLS = [
    "timestep_time",
    "vehicle_id",
    "vehicle_speed",
    "vehicle_accel",
    "vehicle_jerk",
    "vehicle_odometer",
    "vehicle_type",
    "vehicle_x",
    "vehicle_y",
]


@dataclass
class StdReport:
    in_csv: str
    out_csv: str
    rows_in: int
    rows_out: int
    n_vehicles: int
    speed_unit_in: str
    speed_unit_out: str
    speed_converted: bool
    prefer: str
    initial_speed: float
    derived_cols: List[str]
    used_colmap: Dict[str, Optional[str]]
    dt_stats: Dict[str, Optional[float]]
    warnings: List[str]


# ----------------------------- Logging helpers -----------------------------

def setup_logger(log_path: Path, to_console: bool = True) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if to_console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.propagate = False
    return logger


# ----------------------------- Column mapping ------------------------------

def _norm_col(c: str) -> str:
    return c.strip().lower().replace(" ", "_")


def build_alias_map() -> Dict[str, List[str]]:
    """
    Alias candidates (lowercased) for each standard column.

    Notes:
    - vehicle_x/vehicle_y includes: coord_x/coord_y, coordinate_x/coordinate_y (user requested).
    - We use a paired matching strategy for x/y to avoid mismatches.
    """
    return {
        "timestep_time": [
            "timestep_time", "time", "timestamp", "t", "sec", "secs", "second", "seconds"
        ],
        "vehicle_id": [
            "vehicle_id", "veh_id", "vehid", "id", "veh", "vehicle"
        ],
        "vehicle_type": [
            "vehicle_type", "veh_type", "vtype", "type", "class", "vclass"
        ],
        "vehicle_speed": [
            "vehicle_speed", "speed", "v", "vel", "velocity", "veh_speed"
        ],
        "vehicle_accel": [
            "vehicle_accel", "vehicle_acceleration", "accel", "acceleration", "a", "veh_accel"
        ],
        "vehicle_jerk": [
            "vehicle_jerk", "jerk", "j"
        ],
        "vehicle_odometer": [
            "vehicle_odometer", "odometer", "odo", "distance", "dist", "s", "pos", "mileage"
        ],
        # x/y are handled in paired matching; these are fallback singles.
        "vehicle_x": [
            "vehicle_x", "x", "pos_x", "position_x", "location_x",
            "coord_x", "coordinate_x"
        ],
        "vehicle_y": [
            "vehicle_y", "y", "pos_y", "position_y", "location_y",
            "coord_y", "coordinate_y"
        ],
    }


def resolve_xy_pair(cols_norm: Dict[str, str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (src_x, src_y, pair_name).
    Uses priority order for coordinate pairs:
      1) (vehicle_x, vehicle_y)
      2) (coordinate_x, coordinate_y)
      3) (coord_x, coord_y)
      4) (pos_x, pos_y) / (position_x, position_y)
      5) (x, y)
    """
    pairs = [
        ("vehicle_x", "vehicle_y", "vehicle_x/vehicle_y"),
        ("coordinate_x", "coordinate_y", "coordinate_x/coordinate_y"),
        ("coord_x", "coord_y", "coord_x/coord_y"),
        ("pos_x", "pos_y", "pos_x/pos_y"),
        ("position_x", "position_y", "position_x/position_y"),
        ("location_x", "location_y", "location_x/location_y"),
        ("x", "y", "x/y (fallback)"),
    ]
    for xk, yk, name in pairs:
        if xk in cols_norm and yk in cols_norm:
            return cols_norm[xk], cols_norm[yk], name
    return None, None, None


def resolve_columns(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    """
    Resolve/rename columns into standard names.
    Returns df_renamed, used_colmap (std -> source col), warnings.
    """
    warnings: List[str] = []
    alias_map = build_alias_map()

    # Build a mapping from normalized -> original
    cols_norm_to_orig: Dict[str, str] = {_norm_col(c): c for c in df.columns}

    used: Dict[str, Optional[str]] = {k: None for k in STANDARD_COLS}
    rename_map: Dict[str, str] = {}

    # Time, id, type, speed, accel, jerk, odometer: single-column resolution
    singles = ["timestep_time", "vehicle_id", "vehicle_type", "vehicle_speed",
               "vehicle_accel", "vehicle_jerk", "vehicle_odometer"]

    for std in singles:
        for cand in alias_map.get(std, []):
            if cand in cols_norm_to_orig:
                src = cols_norm_to_orig[cand]
                rename_map[src] = std
                used[std] = src
                break

    # XY paired resolution (preferred)
    src_x, src_y, pair_name = resolve_xy_pair(cols_norm_to_orig)
    if src_x and src_y:
        rename_map[src_x] = "vehicle_x"
        rename_map[src_y] = "vehicle_y"
        used["vehicle_x"] = src_x
        used["vehicle_y"] = src_y
        if pair_name and "fallback" in pair_name:
            warnings.append("Using x/y as coordinates (fallback). Please confirm x/y are vehicle coordinates.")
    else:
        # fallback: try single matches
        for std in ["vehicle_x", "vehicle_y"]:
            for cand in alias_map.get(std, []):
                if cand in cols_norm_to_orig:
                    src = cols_norm_to_orig[cand]
                    rename_map[src] = std
                    used[std] = src
                    break
        if used["vehicle_x"] is None or used["vehicle_y"] is None:
            warnings.append("No coordinate (x/y) columns detected. vehicle_x/vehicle_y will be filled with NaN.")

    df2 = df.rename(columns=rename_map)

    # Ensure all standard columns exist (create missing)
    for col in STANDARD_COLS:
        if col not in df2.columns:
            df2[col] = np.nan

    # vehicle_type default
    if df2["vehicle_type"].isna().all():
        df2["vehicle_type"] = "unknown"
        warnings.append("vehicle_type not found; filled with 'unknown'.")

    logger.info("Column mapping used:")
    for k in STANDARD_COLS:
        logger.info(f"  {k:>15s} <= {used.get(k)}")

    for w in warnings:
        logger.warning(w)

    return df2, used, warnings


# ----------------------------- Unit conversion -----------------------------

def convert_speed_to_mps(df: pd.DataFrame, unit_in: str, logger: logging.Logger) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Convert vehicle_speed to m/s if needed.
    unit_in: 'mps', 'm/s', 'kmh', 'km/h', 'mph'
    """
    u = unit_in.strip().lower()
    warnings: List[str] = []
    converted = False

    if "vehicle_speed" not in df.columns:
        return df, False, warnings

    # numeric coercion
    df["vehicle_speed"] = pd.to_numeric(df["vehicle_speed"], errors="coerce")

    factor = 1.0
    if u in ("mps", "m/s", "ms", "m_s"):
        factor = 1.0
    elif u in ("kmh", "km/h", "kph"):
        factor = 1.0 / 3.6
        converted = True
    elif u in ("mph",):
        factor = 0.44704
        converted = True
    else:
        warnings.append(f"Unknown speed unit '{unit_in}'. Assume m/s without conversion.")
        factor = 1.0

    if converted:
        df["vehicle_speed"] = df["vehicle_speed"] * factor
        logger.info(f"Speed unit converted to m/s with factor={factor}. (unit_in={unit_in})")
    else:
        logger.info(f"Speed unit assumed as m/s. (unit_in={unit_in})")

    # light sanity hint (no auto change)
    s = df["vehicle_speed"].dropna()
    if len(s) > 0:
        p95 = float(np.nanpercentile(s.values, 95))
        logger.info(f"Speed p95 (m/s after conversion) = {p95:.3f}")
        if (not converted) and p95 > 60:
            warnings.append("Speed p95 > 60 m/s. This is unusually high; please confirm the input unit.")
        if converted and (u in ("kmh", "km/h", "kph")) and p95 < 10:
            warnings.append("Speed p95 < 10 m/s after km/h->m/s conversion. Please confirm the input unit.")

    for w in warnings:
        logger.warning(w)

    return df, converted, warnings


# ----------------------------- Time hygiene -------------------------------

def sanitize_and_sort(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], List[str]]:
    warnings: List[str] = []
    # Coerce
    df["timestep_time"] = pd.to_numeric(df["timestep_time"], errors="coerce")
    # vehicle_id to string
    df["vehicle_id"] = df["vehicle_id"].astype(str)

    # Drop rows missing essential keys
    before = len(df)
    df = df.dropna(subset=["timestep_time", "vehicle_id"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows with missing timestep_time or vehicle_id.")
        logger.warning(warnings[-1])

    # Sort
    df = df.sort_values(["vehicle_id", "timestep_time"], kind="mergesort").reset_index(drop=True)

    # Compute dt stats globally (informational)
    dt_all = df.groupby("vehicle_id")["timestep_time"].diff()
    dt_pos = dt_all[(dt_all.notna()) & (dt_all > 0)]
    stats = {
        "dt_min": float(dt_pos.min()) if len(dt_pos) else None,
        "dt_median": float(dt_pos.median()) if len(dt_pos) else None,
        "dt_max": float(dt_pos.max()) if len(dt_pos) else None,
        "dt_nonpos_count": int(((dt_all.notna()) & (dt_all <= 0)).sum()),
    }
    logger.info(f"dt stats: {stats}")

    if stats["dt_nonpos_count"] and stats["dt_nonpos_count"] > 0:
        warnings.append("Non-positive dt detected (time not strictly increasing for some vehicles). "
                        "Those steps will be ignored in diff/integration.")
        logger.warning(warnings[-1])

    return df, stats, warnings


# -------------------- Speed-accel consistency & odometer -------------------

def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")


def enforce_speed_accel(df: pd.DataFrame, prefer: str, initial_speed: float,
                        clip_speed_nonneg: bool, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Enforce consistency using only speed and accel (no jerk).
    prefer: 'speed' (default) or 'accel'
    - If accel missing: compute from speed (diff)
    - If speed missing: compute from accel (integral)
    - If both exist:
        prefer='speed' -> recompute accel from speed
        prefer='accel' -> recompute speed from accel
    Returns df, derived_cols, warnings
    """
    prefer = prefer.strip().lower()
    if prefer not in ("speed", "accel"):
        prefer = "speed"

    warnings: List[str] = []
    derived: List[str] = []

    _coerce_numeric(df, "vehicle_speed")
    _coerce_numeric(df, "vehicle_accel")

    def per_vehicle(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestep_time", kind="mergesort").copy()
        t = g["timestep_time"].to_numpy(dtype=float)
        dt = np.diff(t, prepend=np.nan)

        v = g["vehicle_speed"].to_numpy(dtype=float)
        a = g["vehicle_accel"].to_numpy(dtype=float)

        has_v = np.isfinite(v).any()
        has_a = np.isfinite(a).any()

        # Helper: compute accel from speed
        def diff_accel(v_in: np.ndarray) -> np.ndarray:
            a_out = np.full_like(v_in, np.nan, dtype=float)
            for i in range(1, len(v_in)):
                if not (np.isfinite(v_in[i]) and np.isfinite(v_in[i-1]) and np.isfinite(dt[i]) and dt[i] > 0):
                    continue
                a_out[i] = (v_in[i] - v_in[i-1]) / dt[i]
            # first accel -> 0 if missing
            if len(a_out) > 0 and not np.isfinite(a_out[0]):
                a_out[0] = 0.0
            return a_out

        # Helper: integrate speed from accel
        def integ_speed(a_in: np.ndarray) -> np.ndarray:
            v_out = np.full_like(a_in, np.nan, dtype=float)
            # initial value
            v0 = initial_speed if np.isfinite(initial_speed) else 0.0
            v_out[0] = v0
            for i in range(1, len(a_in)):
                if not (np.isfinite(dt[i]) and dt[i] > 0):
                    v_out[i] = v_out[i-1]
                    continue
                ai = a_in[i]
                if not np.isfinite(ai):
                    # hold last speed if accel missing
                    v_out[i] = v_out[i-1]
                    continue
                v_out[i] = v_out[i-1] + ai * dt[i]
                if clip_speed_nonneg and np.isfinite(v_out[i]) and v_out[i] < 0:
                    v_out[i] = 0.0
            return v_out

        # Cases
        if has_v and (not has_a):
            a = diff_accel(v)
            g["vehicle_accel"] = a
            return g

        if has_a and (not has_v):
            v = integ_speed(a)
            g["vehicle_speed"] = v
            return g

        if has_v and has_a:
            if prefer == "speed":
                a = diff_accel(v)
                g["vehicle_accel"] = a
            else:
                v = integ_speed(a)
                g["vehicle_speed"] = v
            return g

        # Neither present: leave as is
        return g

    # Determine which columns will be derived
    any_speed = df["vehicle_speed"].notna().any()
    any_accel = df["vehicle_accel"].notna().any()

    if any_speed and (not any_accel):
        derived.append("vehicle_accel (diff from vehicle_speed)")
        logger.info("vehicle_accel missing -> will derive from vehicle_speed (diff).")
    elif any_accel and (not any_speed):
        derived.append("vehicle_speed (integral from vehicle_accel)")
        logger.info("vehicle_speed missing -> will derive from vehicle_accel (integral).")
    elif any_speed and any_accel:
        if prefer == "speed":
            derived.append("vehicle_accel (recomputed from vehicle_speed due to prefer=speed)")
        else:
            derived.append("vehicle_speed (recomputed from vehicle_accel due to prefer=accel)")
    else:
        warnings.append("Both vehicle_speed and vehicle_accel are missing (cannot enforce consistency).")
        logger.warning(warnings[-1])

    df2 = df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)

    return df2, derived, warnings


def compute_odometer(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    If vehicle_odometer missing, compute by integrating speed over dt per vehicle.
    """
    warnings: List[str] = []
    derived: List[str] = []

    _coerce_numeric(df, "vehicle_odometer")
    _coerce_numeric(df, "vehicle_speed")

    has_odo = df["vehicle_odometer"].notna().any()
    if has_odo:
        logger.info("vehicle_odometer exists in input -> keep (no integration).")
        return df, derived, warnings

    derived.append("vehicle_odometer (integral of vehicle_speed over dt)")
    logger.info("vehicle_odometer missing -> derive from vehicle_speed integration.")

    def per_vehicle(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestep_time", kind="mergesort").copy()
        t = g["timestep_time"].to_numpy(dtype=float)
        dt = np.diff(t, prepend=np.nan)
        v = g["vehicle_speed"].to_numpy(dtype=float)

        odo = np.full_like(v, np.nan, dtype=float)
        odo[0] = 0.0
        for i in range(1, len(v)):
            if not (np.isfinite(dt[i]) and dt[i] > 0 and np.isfinite(odo[i-1])):
                odo[i] = odo[i-1]
                continue
            vi = v[i]
            if not np.isfinite(vi):
                odo[i] = odo[i-1]
                continue
            odo[i] = odo[i-1] + vi * dt[i]
        g["vehicle_odometer"] = odo
        return g

    df2 = df.groupby("vehicle_id", group_keys=False).apply(per_vehicle)
    return df2, derived, warnings


# ----------------------------- Main standardizer ---------------------------

def standardize_ground_truth(
    in_csv: Path,
    out_dir: Path,
    speed_unit_in: str = "m/s",
    prefer: str = "speed",
    initial_speed: float = 0.0,
    clip_speed_nonneg: bool = True,
    keep_input_jerk: bool = False,
    encoding: str = "utf-8-sig",
    to_console: bool = True,
) -> StdReport:
    out_dir = out_dir / "00_ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"ground_truth_std_{ts}.log"
    logger = setup_logger(log_path, to_console=to_console)

    logger.info("=== Ground Truth Standardization شروع ===")
    logger.info(f"in_csv  : {in_csv}")
    logger.info(f"out_dir : {out_dir}")
    logger.info(f"speed_unit_in: {speed_unit_in} (default m/s)")
    logger.info(f"prefer  : {prefer}")
    logger.info(f"initial_speed (for accel->speed integral): {initial_speed}")
    logger.info(f"clip_speed_nonneg: {clip_speed_nonneg}")
    logger.info(f"keep_input_jerk: {keep_input_jerk}")
    logger.info(f"encoding: {encoding}")

    # Read
    df_raw = pd.read_csv(in_csv, encoding=encoding)
    rows_in = len(df_raw)
    logger.info(f"Read rows: {rows_in}, cols: {len(df_raw.columns)}")

    # Resolve columns
    df, used_colmap, warn_cols = resolve_columns(df_raw, logger)

    # Speed unit conversion to m/s
    df, converted, warn_unit = convert_speed_to_mps(df, speed_unit_in, logger)

    # Time hygiene + sort
    df, dt_stats, warn_dt = sanitize_and_sort(df, logger)

    # Essential checks
    # Must have at least one of speed/accel to proceed meaningfully
    has_speed = df["vehicle_speed"].notna().any()
    has_accel = df["vehicle_accel"].notna().any()
    if not (has_speed or has_accel):
        raise ValueError("Input does not contain vehicle_speed or vehicle_accel (or any recognized aliases).")

    # Enforce speed-accel consistency (NO jerk)
    df, derived_sa, warn_sa = enforce_speed_accel(
        df=df,
        prefer=prefer,
        initial_speed=initial_speed,
        clip_speed_nonneg=clip_speed_nonneg,
        logger=logger,
    )

    # Odometer
    df, derived_odo, warn_odo = compute_odometer(df, logger)

    # Jerk placeholder
    derived_cols = []
    derived_cols.extend(derived_sa)
    derived_cols.extend(derived_odo)

    if keep_input_jerk and used_colmap.get("vehicle_jerk") is not None:
        logger.info("Keeping input jerk column as-is (no computation).")
    else:
        # Do not compute jerk; keep placeholder NaN
        df["vehicle_jerk"] = np.nan
        if used_colmap.get("vehicle_jerk") is not None:
            logger.info("Input jerk found but overwritten to NaN because keep_input_jerk=False (default).")
        derived_cols.append("vehicle_jerk (placeholder NaN; jerk not computed)")

    # Ensure final columns and order
    for c in STANDARD_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[STANDARD_COLS].copy()

    # Final sort
    df = df.sort_values(["vehicle_id", "timestep_time"], kind="mergesort").reset_index(drop=True)

    # Stats & print
    n_vehicles = df["vehicle_id"].nunique()
    rows_out = len(df)

    logger.info(f"Vehicles detected: {n_vehicles}")
    logger.info(f"Output rows: {rows_out}")
    logger.info("Derived columns:")
    for d in derived_cols:
        logger.info(f"  - {d}")

    warnings = []
    warnings.extend(warn_cols)
    warnings.extend(warn_unit)
    warnings.extend(warn_dt)
    warnings.extend(warn_sa)
    warnings.extend(warn_odo)

    # Write outputs
    out_csv = out_dir / "ground_truth_std.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved standardized CSV: {out_csv}")

    report = StdReport(
        in_csv=str(in_csv),
        out_csv=str(out_csv),
        rows_in=rows_in,
        rows_out=rows_out,
        n_vehicles=int(n_vehicles),
        speed_unit_in=speed_unit_in,
        speed_unit_out="m/s",
        speed_converted=bool(converted),
        prefer=prefer,
        initial_speed=float(initial_speed),
        derived_cols=derived_cols,
        used_colmap=used_colmap,
        dt_stats=dt_stats,
        warnings=warnings,
    )
    out_report = out_dir / "ground_truth_std_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    logger.info(f"Saved report JSON: {out_report}")

    # User-facing summary (console)
    print("\n=== Ground Truth Standardization Summary ===")
    print(f"Input : {in_csv}")
    print(f"Output: {out_csv}")
    print(f"Vehicles: {n_vehicles} | Rows: {rows_out}")
    if converted:
        print(f"Speed unit converted: {speed_unit_in} -> m/s")
    else:
        print(f"Speed unit assumed: m/s (unit_in={speed_unit_in})")
    print(f"Prefer consistency: {prefer}")
    if derived_cols:
        print("Derived:")
        for d in derived_cols:
            print(f"  - {d}")
    if warnings:
        print("Warnings:")
        for w in warnings[:12]:
            print(f"  - {w}")
        if len(warnings) > 12:
            print(f"  ... ({len(warnings)-12} more)")
    print("===========================================\n")

    return report


# ----------------------------- CLI entry -----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize ground-truth CSV to project column conventions.")
    p.add_argument("--in_csv", required=True, help="Path to input ground truth CSV.")
    p.add_argument("--out_dir", required=True, help="Output root directory. Will create 00_ground_truth/ inside.")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig).")

    p.add_argument("--speed_unit_in", default="m/s",
                   help="Input speed unit: m/s (default), km/h, mph. Output is always m/s.")
    p.add_argument("--prefer", default="speed", choices=["speed", "accel"],
                   help="When both speed and accel exist, which one to trust for consistency (default: speed).")
    p.add_argument("--initial_speed", type=float, default=0.0,
                   help="Initial speed (m/s) when reconstructing speed by integrating accel (default: 0).")

    p.add_argument("--no_clip_speed_nonneg", action="store_true",
                   help="Disable clipping negative reconstructed speed to 0.")
    p.add_argument("--keep_input_jerk", action="store_true",
                   help="Keep input jerk column if present (still will not compute jerk).")
    p.add_argument("--no_console_log", action="store_true",
                   help="Disable console logging (file log still saved).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.in_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not in_csv.exists() or not in_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    standardize_ground_truth(
        in_csv=in_csv,
        out_dir=out_dir,
        speed_unit_in=args.speed_unit_in,
        prefer=args.prefer,
        initial_speed=args.initial_speed,
        clip_speed_nonneg=(not args.no_clip_speed_nonneg),
        keep_input_jerk=args.keep_input_jerk,
        encoding=args.encoding,
        to_console=(not args.no_console_log),
    )


if __name__ == "__main__":
    main()

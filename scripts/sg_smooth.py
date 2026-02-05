# -*- coding: utf-8 -*-
"""
baseline_smooth_export_core_cols.py

Final output columns (ONLY):
  timestep_time, vehicle_id, trip_id, vehicle_speed, vehicle_accel, vehicle_jerk, vehicle_odometer

Behavior:
- Stage A (parallel): smooth speed per (trip_id, vehicle_id), accel clip, distance preserve
- Stage B (single): rebuild continuous odometer per vehicle_id by integrating smoothed speed
- Overwrite:
    vehicle_speed  <- smoothed speed
    vehicle_accel  <- smoothed accel (diff of smoothed speed)
    vehicle_odometer <- rebuilt odometer
- Compute jerk per vehicle_id:
    vehicle_jerk = diff(vehicle_accel)/diff(timestep_time)
- Drop all other columns and save CSV.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ----------------------------- Config -----------------------------

@dataclass(frozen=True)
class BaselineConfig:
    # endpoint zero lock
    zero_eps: float = 1e-6

    # Savitzky-Golay
    sg_window: int = 11
    sg_poly: int = 3
    min_len_for_sg: int = 5

    # bounds
    v_min: float = 0.0
    v_max: float = 16.67      # 60 km/h
    a_min: float = -4.5
    a_max: float = 2.5

    # numeric
    eps_div: float = 1e-12

    # group validity
    min_valid_points: int = 6

    # parallel
    default_workers_keep_free: int = 1  # cpu_count - keep_free


# ----------------------------- Logging -----------------------------

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("baseline")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


# ----------------------------- Utilities -----------------------------

def nearest_odd(n: int) -> int:
    n = int(n)
    if n < 1:
        return 1
    return n if n % 2 == 1 else n - 1


def clip_arr(x: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    y = x
    if lo is not None:
        y = np.maximum(y, lo)
    if hi is not None:
        y = np.minimum(y, hi)
    return y


def safe_dt(t: np.ndarray) -> np.ndarray:
    """
    dt[k] = t[k+1] - t[k]
    If dt is non-positive or not finite, replace by median(valid_dt) or 1.0.
    """
    dt = np.diff(t)
    ok = np.isfinite(dt) & (dt > 0)
    dt_med = float(np.median(dt[ok])) if ok.any() else 1.0
    return np.where(ok, dt, dt_med)


def sg_smooth(v: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    n = len(v)
    if n < cfg.min_len_for_sg:
        return v.copy()

    w = nearest_odd(min(cfg.sg_window, n if n % 2 == 1 else n - 1))
    if w < cfg.min_len_for_sg:
        return v.copy()

    p = min(cfg.sg_poly, w - 2)  # ensure p < w
    if p < 1:
        return v.copy()

    return savgol_filter(v, window_length=w, polyorder=p, mode="interp")


def accel_clip_and_reintegrate(v: np.ndarray, t: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    """
    a = diff(v)/dt -> clip(a) -> re-integrate to v_rec to ensure v-a consistency.
    """
    if len(v) < 2:
        return v.copy()

    dt = safe_dt(t)
    a = np.diff(v) / dt
    a = clip_arr(a, cfg.a_min, cfg.a_max)

    v_rec = np.empty_like(v)
    v_rec[0] = v[0]
    for k in range(len(a)):
        v_rec[k + 1] = v_rec[k] + a[k] * dt[k]

    return clip_arr(v_rec, cfg.v_min, cfg.v_max)


def enforce_zero_endpoints_if_needed(v_final: np.ndarray, v_raw: np.ndarray, cfg: BaselineConfig) -> None:
    if len(v_final) == 0:
        return
    if np.isfinite(v_raw[0]) and abs(v_raw[0]) <= cfg.zero_eps:
        v_final[0] = 0.0
    if np.isfinite(v_raw[-1]) and abs(v_raw[-1]) <= cfg.zero_eps:
        v_final[-1] = 0.0


def distance_preserve_scale_interior(
    v: np.ndarray,
    t: np.ndarray,
    odo: np.ndarray,
    cfg: BaselineConfig,
) -> tuple[np.ndarray, float]:
    """
    Target distance: S_target = odo[-1] - odo[0]
    Scale ONLY interior points v[1:-1] so that interior distance matches target.
    """
    n = len(v)
    if n < 2:
        return v.copy(), 1.0

    dt = safe_dt(t)

    if not (np.isfinite(odo[0]) and np.isfinite(odo[-1])):
        return v.copy(), 1.0

    S_target = float(odo[-1] - odo[0])
    if (not np.isfinite(S_target)) or (S_target < 0):
        return v.copy(), 1.0

    if n <= 3:
        return v.copy(), 1.0

    S_inner = float(np.sum(v[1:-1] * dt[1:]))  # align with dt[1:]
    lam = S_target / (S_inner + cfg.eps_div)

    v2 = v.copy()
    v2[1:-1] = clip_arr(v2[1:-1] * lam, cfg.v_min, cfg.v_max)

    return v2, float(lam)


def compute_accel(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    accel aligned to each time point:
      a[i] = (v[i] - v[i-1]) / (t[i] - t[i-1]) for i>=1
      a[0] = 0
    """
    n = len(v)
    a = np.zeros(n, dtype=float)
    if n < 2:
        return a
    dt = np.diff(t)
    ok = np.isfinite(dt) & (dt > 0)
    dt_med = float(np.median(dt[ok])) if ok.any() else 1.0
    dt_safe = np.where(ok, dt, dt_med)

    dv = np.diff(v)
    a[1:] = dv / dt_safe
    a = clip_arr(a, None, None)  # no extra clip here; already clipped via reintegration
    return a


# ----------------------------- Column handling -----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert legacy column names to standard English names.
    - 工况ID -> trip_id
    """
    df = df.copy()
    rename_map = {}
    if "trip_id" not in df.columns and "工况ID" in df.columns:
        rename_map["工况ID"] = "trip_id"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ----------------------------- Core processing -----------------------------

def smooth_group(g: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    """
    Process one (trip_id, vehicle_id) group and return with temporary smoothed columns.
    """
    g = g.sort_values("timestep_time").copy()

    t = g["timestep_time"].to_numpy(dtype=float)
    v_raw = g["vehicle_speed"].to_numpy(dtype=float)
    odo = g["vehicle_odometer"].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(v_raw) & np.isfinite(odo)
    if valid.sum() < cfg.min_valid_points:
        # fallback: keep raw speed; accel from raw
        v_fin = clip_arr(v_raw, cfg.v_min, cfg.v_max)
        a_fin = compute_accel(v_fin, t)
        g["_sm_speed"] = v_fin
        g["_sm_accel"] = a_fin
        return g

    # 1) SG smooth
    v_sg = sg_smooth(v_raw, cfg)

    # 2) speed clip
    v_clip = clip_arr(v_sg, cfg.v_min, cfg.v_max)

    # 3) accel clip + reintegrate
    v_rec = accel_clip_and_reintegrate(v_clip, t, cfg)

    # 4) endpoint zero lock if needed
    enforce_zero_endpoints_if_needed(v_rec, v_raw, cfg)

    # 5) preserve distance by scaling interior points
    v_fin, _ = distance_preserve_scale_interior(v_rec, t, odo, cfg)

    enforce_zero_endpoints_if_needed(v_fin, v_raw, cfg)

    # accel from final speed
    a_fin = compute_accel(v_fin, t)

    g["_sm_speed"] = v_fin
    g["_sm_accel"] = a_fin
    return g


def run_stage_a(df: pd.DataFrame, cfg: BaselineConfig, max_workers: int | None, logger: logging.Logger) -> pd.DataFrame:
    require_columns(df, ["trip_id", "vehicle_id", "timestep_time", "vehicle_speed", "vehicle_odometer"])
    df = df.sort_values(["trip_id", "vehicle_id", "timestep_time"]).copy()

    if max_workers is None or max_workers <= 0:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - cfg.default_workers_keep_free)

    groups = [g for _, g in df.groupby(["trip_id", "vehicle_id"], sort=False)]
    logger.info(f"Stage A: groups={len(groups)}, max_workers={max_workers}")

    parts: list[pd.DataFrame] = []
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(smooth_group, g, cfg) for g in groups]
        for fu in as_completed(futs):
            try:
                parts.append(fu.result())
            except Exception as e:
                failed += 1
                logger.error(f"Stage A group failed: {repr(e)}")

    if not parts:
        raise RuntimeError("Stage A produced no results (all groups failed?).")

    out = pd.concat(parts, axis=0).sort_index()
    if failed > 0:
        logger.warning(f"Stage A finished with {failed} failed groups.")
    return out


def run_stage_b_rebuild_odometer(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Rebuild continuous odometer per vehicle_id by integrating smoothed speed.
    Use raw vehicle_odometer at the first timestamp per vehicle as the starting point.
    """
    require_columns(df, ["vehicle_id", "timestep_time", "vehicle_odometer", "_sm_speed"])
    df = df.sort_values(["vehicle_id", "timestep_time"]).copy()

    out_odo = np.full(len(df), np.nan, dtype=float)

    for vid, g in df.groupby("vehicle_id", sort=False):
        idx = g.index.to_numpy()
        t = g["timestep_time"].to_numpy(dtype=float)
        v = g["_sm_speed"].to_numpy(dtype=float)

        if len(g) == 0:
            continue

        odo0_raw = g["vehicle_odometer"].iloc[0]
        odo0 = float(odo0_raw) if np.isfinite(odo0_raw) else 0.0

        odo_new = np.full(len(g), np.nan, dtype=float)
        odo_new[0] = odo0

        if len(g) >= 2:
            dt = safe_dt(t)
            for k in range(len(dt)):
                vv = v[k] if np.isfinite(v[k]) else 0.0
                odo_new[k + 1] = odo_new[k] + vv * dt[k]

        out_odo[idx] = odo_new

    df["_sm_odometer"] = out_odo
    logger.info("Stage B finished (odometer rebuilt).")
    return df


def compute_jerk_per_vehicle(df: pd.DataFrame) -> pd.Series:
    """
    jerk[i] = (accel[i] - accel[i-1]) / (t[i] - t[i-1]) within each vehicle_id.
    jerk for first row of each vehicle is 0.
    """
    df_sorted = df.sort_values(["vehicle_id", "timestep_time"])
    jerk = pd.Series(index=df_sorted.index, dtype=float)

    for vid, g in df_sorted.groupby("vehicle_id", sort=False):
        t = g["timestep_time"].to_numpy(dtype=float)
        a = g["vehicle_accel"].to_numpy(dtype=float)

        j = np.zeros(len(g), dtype=float)
        if len(g) >= 2:
            dt = np.diff(t)
            ok = np.isfinite(dt) & (dt > 0)
            dt_med = float(np.median(dt[ok])) if ok.any() else 1.0
            dt_safe = np.where(ok, dt, dt_med)

            da = np.diff(a)
            j[1:] = da / dt_safe

        jerk.loc[g.index] = j

    # restore original index order alignment
    return jerk.reindex(df.index)


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline smoothing (parallel) and export only core columns.")
    p.add_argument("--in_csv", required=True, help="Input CSV path.")
    p.add_argument("--out_csv", required=True, help="Output CSV path.")
    p.add_argument("--max_workers", type=int, default=0, help="Process workers for Stage A. 0=auto.")
    p.add_argument("--log_level", default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)
    cfg = BaselineConfig()

    logger.info(f"Reading: {args.in_csv}")
    df = pd.read_csv(args.in_csv)
    df = standardize_columns(df)

    require_columns(df, ["timestep_time", "vehicle_id", "trip_id", "vehicle_speed", "vehicle_odometer"])

    # Stage A: speed smoothing + accel
    df_a = run_stage_a(df, cfg=cfg, max_workers=args.max_workers, logger=logger)

    # Stage B: rebuild continuous odometer based on smoothed speed
    df_b = run_stage_b_rebuild_odometer(df_a, logger=logger)

    # Overwrite original columns with smoothed results
    df_b["vehicle_speed"] = df_b["_sm_speed"].astype(float)
    df_b["vehicle_accel"] = df_b["_sm_accel"].astype(float).fillna(0.0)
    df_b["vehicle_odometer"] = df_b["_sm_odometer"].astype(float)

    # Compute jerk from overwritten accel
    df_b["vehicle_jerk"] = compute_jerk_per_vehicle(df_b).astype(float).fillna(0.0)

    # Final column selection ONLY
    final_cols = [
        "timestep_time",
        "vehicle_id",
        "trip_id",
        "vehicle_speed",
        "vehicle_accel",
        "vehicle_jerk",
        "vehicle_odometer",
    ]

    df_out = df_b[final_cols].copy()

    # sort for stable output (recommended)
    df_out.sort_values(["timestep_time", "vehicle_id"], inplace=True)

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved: {args.out_csv}")
    logger.info(f"Output columns: {final_cols}")


if __name__ == "__main__":
    main()

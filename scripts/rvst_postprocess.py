# -*- coding: utf-8 -*-
"""
rvst_postprocess.py

Postprocess for RVST:
1) filter_bad_cases_by_jerk_envelope: find bad (trip_id, vehicle_id) cases based on jerk envelope
2) merge_stage1_stage2: overwrite stage1 by stage2 where stage2 has values

v4 update (this file):
- Filter logic aligned with user's original rule:
  * sort by (trip_id, vehicle_id, timestep_time)
  * jerk_t = accel_{t+1} - accel_t (optionally /dt)
  * merge jerk envelope by accel_round
  * point in-envelope if jerk and threshold exist and within [min,max]
  * case-level decision:
      - compute idx_in_case on jerk-valid points (exclude last jerk NaN)
      - ONLY check middle points idx=1..n-2
      - if n_points < 3 => pass True (no middle points)
  * bad_ids / ok_ids are case keys: (trip_id, vehicle_id)
  * export BadCase_ids.csv with BOTH columns (trip_id, vehicle_id)

Logging:
- default file only, console off unless log_to_console=True
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# =========================================================
@dataclass
class OutPaths:
    root: str
    badcase_dir: str
    merge_dir: str
    logs_dir: str

    @staticmethod
    def make(out_dir: str) -> "OutPaths":
        root = os.path.abspath(out_dir)
        return OutPaths(
            root=root,
            badcase_dir=os.path.join(root, "15_badcase"),
            merge_dir=os.path.join(root, "99_merge"),
            logs_dir=os.path.join(root, "logs"),
        )

    def ensure(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.badcase_dir, exist_ok=True)
        os.makedirs(self.merge_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


# =========================================================
def setup_logger(log_path: str, level=logging.INFO, to_console: bool = False) -> logging.Logger:
    logger = logging.getLogger("rvst_postprocess")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # v4 default: no console handler
    if to_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def sanitize_filename(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required columns:{miss}")


def resolve_resource_path(user_path: Optional[str], filename: str) -> str:
    if user_path:
        p = os.path.abspath(user_path)
        if os.path.exists(p):
            return p

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "resources", filename),
        os.path.join(cwd, filename),
        os.path.join(cwd, "resources", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Resource file not found: {filename}\n"
        f"Tried:\n- " + "\n- ".join(candidates) + "\n\n"
        f"How to fix:\n"
        f"1) Put {filename} next to this script or under resources/\n"
        f"2) Or pass --jerk_env_csv \"<path>\" explicitly"
    )


def _force_keys_post(df: pd.DataFrame, col_vid: str, col_t: str, col_case: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if col_case and col_case in df.columns:
        df[col_case] = df[col_case].astype(str)
    if col_vid in df.columns:
        df[col_vid] = df[col_vid].astype(str)
    if col_t in df.columns:
        df[col_t] = pd.to_numeric(df[col_t], errors="coerce")
    return df


# =========================================================
@dataclass
class FilterConfig:
    stage1_csv: str
    out_dir: str
    prefix: str = "fcd"
    jerk_env_csv: str = ""
    accel_round_decimals: int = 1
    jerk_use_dt: bool = False
    dt: float = 1.0
    export_points: bool = True
    csv_encoding: str = "utf-8-sig"
    log_to_console: bool = False

    col_case: str = "trip_id"
    col_vid: str = "vehicle_id"
    col_t: str = "timestep_time"
    col_opt_a: str = "vehicle_accel"
    col_opt_v: str = "vehicle_speed"
    col_opt_x: str = "vehicle_odometer"


@dataclass
class MergeConfig:
    stage1_csv: str
    stage2_csv: str
    out_dir: str
    prefix: str = "fcd"
    out_name: str = "rvst_all"
    replace_policy: str = "notna_any"     # "notna_any" | "notna_all3"
    csv_encoding: str = "utf-8-sig"
    log_to_console: bool = False

    col_vid: str = "vehicle_id"
    col_t: str = "timestep_time"


# =========================================================
def load_jerk_envelope(jerk_env_csv: str, decimals: int, csv_encoding: str) -> pd.DataFrame:
    env = pd.read_csv(jerk_env_csv, encoding=csv_encoding)
    require_cols(env, ["accel_round", "q1_smooth", "q99_smooth"], "jerk_env_csv")

    env = env.copy()
    env["accel_round"] = pd.to_numeric(env["accel_round"], errors="coerce").round(decimals)
    env["jerk_min"] = pd.to_numeric(env["q1_smooth"], errors="coerce")
    env["jerk_max"] = pd.to_numeric(env["q99_smooth"], errors="coerce")
    env = env.dropna(subset=["accel_round"]).copy()

    # If duplicates exist for accel_round, keep conservative bounds
    env = env.groupby("accel_round", as_index=False).agg(
        jerk_min=("jerk_min", "min"),
        jerk_max=("jerk_max", "max"),
    )
    return env


# =========================================================
def filter_bad_cases_by_jerk_envelope(cfg: FilterConfig) -> Tuple[str, str]:
    """
    Output:
      - BadCase.csv: all stage1 rows for bad (trip_id, vehicle_id) cases
      - BadCase_summary.csv: case-level summary (ok/bad + stats)
      - OkCase_ids.csv / BadCase_ids.csv: (trip_id, vehicle_id) ids
      - (optional) BadCase_points.csv: point-level diagnostic for jerk points
    """
    out_paths = OutPaths.make(cfg.out_dir)
    out_paths.ensure()
    logger = setup_logger(
        os.path.join(out_paths.logs_dir, "rvst_filter_badcase.log"),
        to_console=cfg.log_to_console
    )

    t0 = time.time()
    logger.info("=== RVST Filter: Filter Bad Cases by Jerk Envelope (mid-points only) ===")
    logger.info(f"stage1_csv: {cfg.stage1_csv}")
    logger.info(f"out_dir   : {out_paths.root}")

    jerk_env_path = resolve_resource_path(cfg.jerk_env_csv, "jerk_envelope.csv")
    logger.info(f"jerk_env  : {jerk_env_path}")

    df = pd.read_csv(cfg.stage1_csv, encoding=cfg.csv_encoding)
    require_cols(df, [cfg.col_case, cfg.col_vid, cfg.col_t, cfg.col_opt_a, cfg.col_opt_v, cfg.col_opt_x], "stage1_csv")

    # dtype hygiene
    df = _force_keys_post(df, cfg.col_vid, cfg.col_t, cfg.col_case)
    df = df.dropna(subset=[cfg.col_t]).copy()

    # keep rows where speed is not NaN (same as your original logic)
    df = df[df[cfg.col_opt_v].notna()].copy()

    # stable sort: trip_id, vehicle_id, timestep_time
    df = df.sort_values([cfg.col_case, cfg.col_vid, cfg.col_t], kind="mergesort").reset_index(drop=True)

    # accel numeric
    df["_a_num"] = pd.to_numeric(df[cfg.col_opt_a], errors="coerce")

    # jerk forward diff within each (trip_id, vehicle_id): jerk_t = a_{t+1} - a_t assigned to row t
    gcols = [cfg.col_case, cfg.col_vid]
    df["_a_next"] = df.groupby(gcols)["_a_num"].shift(-1)
    df["_jerk"] = df["_a_next"] - df["_a_num"]
    if cfg.jerk_use_dt and cfg.dt != 0:
        df["_jerk"] = df["_jerk"] / float(cfg.dt)

    # jerk-valid points (exclude last row of each case where a_next is NaN)
    df_points = df.dropna(subset=["_jerk"]).copy()

    # accel round for envelope match
    df_points["_accel_round"] = pd.to_numeric(df_points["_a_num"], errors="coerce").round(cfg.accel_round_decimals)

    # load envelope + merge
    env = load_jerk_envelope(jerk_env_path, cfg.accel_round_decimals, cfg.csv_encoding)
    df_points = df_points.merge(env, left_on="_accel_round", right_on="accel_round", how="left")

    # point-level in-envelope
    df_points["_has_thr"] = df_points["jerk_min"].notna() & df_points["jerk_max"].notna()
    df_points["_in_env"] = (
        df_points["_has_thr"]
        & (df_points["_jerk"] >= df_points["jerk_min"])
        & (df_points["_jerk"] <= df_points["jerk_max"])
    )

    # ------------------------------------------------------------
    # Case-level decision: ONLY check middle points idx=1..n-2 on jerk-valid points
    # ------------------------------------------------------------
    df_points = df_points.sort_values([cfg.col_case, cfg.col_vid, cfg.col_t], kind="mergesort").copy()
    df_points["n_in_case"] = df_points.groupby(gcols)[cfg.col_case].transform("size")
    df_points["idx_in_case"] = df_points.groupby(gcols).cumcount()

    # middle points
    df_mid = df_points[
        (df_points["idx_in_case"] >= 1) & (df_points["idx_in_case"] <= df_points["n_in_case"] - 2)
    ].copy()

    # all middle points must be in envelope
    case_all_mid = df_mid.groupby(gcols)["_in_env"].all()

    # build all case index from df (not df_points) to cover cases even if jerk points are empty
    all_case_df = df[gcols].drop_duplicates().copy()
    all_case_df[cfg.col_case] = all_case_df[cfg.col_case].astype(str)
    all_case_df[cfg.col_vid] = all_case_df[cfg.col_vid].astype(str)

    all_case_tuples = list(zip(all_case_df[cfg.col_case], all_case_df[cfg.col_vid]))
    all_case_index = pd.MultiIndex.from_tuples(all_case_tuples, names=gcols)

    case_all_mid = case_all_mid.reindex(all_case_index, fill_value=False)

    # if jerk-valid points < 3 => no middle points, set True (same as your original script)
    # Note: n_in_case is defined on df_points (jerk-valid points).
    # For cases absent in df_points => treat as small and set True (conservative per your original rule).
    npts = df_points.groupby(gcols).size()
    npts = npts.reindex(all_case_index, fill_value=0)
    small_mask = (npts < 3)
    case_all_mid.loc[small_mask] = True

    ok_mask = case_all_mid.astype(bool)
    ok_ids_df = pd.DataFrame(ok_mask[ok_mask].index.tolist(), columns=gcols)
    bad_ids_df = pd.DataFrame(ok_mask[~ok_mask].index.tolist(), columns=gcols)

    # summary stats at case level (still useful for diagnosis)
    df_points["_missing_thr"] = (~df_points["_has_thr"]).astype(int)
    df_points["_out_range"] = ((df_points["_has_thr"]) & (~df_points["_in_env"])).astype(int)

    case_summary = (
        df_points.groupby(gcols, as_index=False)
        .agg(
            n_jerk_points=("_in_env", "size"),
            n_in_env=("_in_env", "sum"),
            n_missing_thr=("_missing_thr", "sum"),
            n_out_of_range=("_out_range", "sum"),
            jerk_min_obs=("_jerk", "min"),
            jerk_max_obs=("_jerk", "max"),
        )
    )

    # attach pass flag (aligned with all_case_index)
    pass_df = pd.DataFrame(ok_mask.reset_index())
    pass_df.columns = gcols + ["pass_in_env_mid"]
    case_summary = pass_df.merge(case_summary, on=gcols, how="left")

    # also include cases that have no jerk points
    if len(case_summary) < len(all_case_df):
        case_summary = all_case_df.merge(case_summary, on=gcols, how="left")
        case_summary["pass_in_env_mid"] = case_summary["pass_in_env_mid"].fillna(False)
        case_summary["n_jerk_points"] = case_summary["n_jerk_points"].fillna(0).astype(int)
        case_summary["n_in_env"] = case_summary["n_in_env"].fillna(0).astype(int)
        case_summary["n_missing_thr"] = case_summary["n_missing_thr"].fillna(0).astype(int)
        case_summary["n_out_of_range"] = case_summary["n_out_of_range"].fillna(0).astype(int)

    # collect bad rows from original df (all timesteps)
    bad_key = set(zip(bad_ids_df[cfg.col_case].astype(str), bad_ids_df[cfg.col_vid].astype(str)))
    df["_case_key"] = list(zip(df[cfg.col_case].astype(str), df[cfg.col_vid].astype(str)))
    bad_rows = df[df["_case_key"].isin(bad_key)].drop(columns=["_case_key"], errors="ignore").copy()

    # outputs
    prefix_safe = sanitize_filename(cfg.prefix)

    badcase_csv = os.path.join(out_paths.badcase_dir, f"{prefix_safe}_BadCase.csv")
    summary_csv = os.path.join(out_paths.badcase_dir, f"{prefix_safe}_BadCase_summary.csv")
    ok_ids_csv = os.path.join(out_paths.badcase_dir, f"{prefix_safe}_OkCase_ids.csv")
    bad_ids_csv = os.path.join(out_paths.badcase_dir, f"{prefix_safe}_BadCase_ids.csv")

    bad_rows.to_csv(badcase_csv, index=False, encoding=cfg.csv_encoding)
    case_summary.to_csv(summary_csv, index=False, encoding=cfg.csv_encoding)
    ok_ids_df.to_csv(ok_ids_csv, index=False, encoding=cfg.csv_encoding)
    bad_ids_df.to_csv(bad_ids_csv, index=False, encoding=cfg.csv_encoding)

    if cfg.export_points:
        points_csv = os.path.join(out_paths.badcase_dir, f"{prefix_safe}_BadCase_points.csv")
        keep_cols = [
            cfg.col_case, cfg.col_vid, cfg.col_t,
            cfg.col_opt_a, "_a_next", "_jerk", "_accel_round",
            "jerk_min", "jerk_max", "_has_thr", "_in_env",
            "idx_in_case", "n_in_case"
        ]
        keep_cols = [c for c in keep_cols if c in df_points.columns]
        df_points[keep_cols].to_csv(points_csv, index=False, encoding=cfg.csv_encoding)

    snapshot = {
        "cfg": asdict(cfg),
        "jerk_env_csv_resolved": jerk_env_path,
        "rule": {
            "point_in_env": "has_thr & jerk in [min,max]",
            "case_pass": "ONLY middle points idx=1..n-2 on jerk-valid points must all be in_env; n<3 => pass True",
            "case_key": f"({cfg.col_case}, {cfg.col_vid})",
        },
        "stats": {
            "total_cases": int(len(all_case_df)),
            "bad_cases": int(len(bad_ids_df)),
            "ok_cases": int(len(ok_ids_df)),
            "bad_rows": int(len(bad_rows)),
            "jerk_points": int(len(df_points)),
            "mid_points": int(len(df_mid)),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(out_paths.badcase_dir, f"{prefix_safe}_BadCase_snapshot.json"), snapshot)

    logger.info(f"time: {(time.time()-t0):.2f} sec")
    return badcase_csv, summary_csv


# =========================================================
def merge_stage1_stage2(cfg: MergeConfig) -> str:
    out_paths = OutPaths.make(cfg.out_dir)
    out_paths.ensure()
    logger = setup_logger(
        os.path.join(out_paths.logs_dir, "rvst_merge_stage1_stage2.log"),
        to_console=cfg.log_to_console
    )

    t0 = time.time()
    prefix_safe = sanitize_filename(cfg.prefix)

    logger.info("=== RVST Merge: Stage2 overwrite Stage1 ===")
    logger.info(f"stage1_csv: {cfg.stage1_csv}")
    logger.info(f"stage2_csv: {cfg.stage2_csv}")
    logger.info(f"out_dir   : {out_paths.root}")
    logger.info(f"policy    : {cfg.replace_policy}")

    df1 = pd.read_csv(cfg.stage1_csv, encoding=cfg.csv_encoding)
    df2 = pd.read_csv(cfg.stage2_csv, encoding=cfg.csv_encoding)

    require_cols(df1, [cfg.col_vid, cfg.col_t], "stage1_csv")
    require_cols(df2, [cfg.col_vid, cfg.col_t], "stage2_csv")

    # dtype hygiene BEFORE determining key/merge
    has_trip = ("trip_id" in df1.columns and "trip_id" in df2.columns)
    df1 = _force_keys_post(df1, cfg.col_vid, cfg.col_t, "trip_id" if has_trip else None)
    df2 = _force_keys_post(df2, cfg.col_vid, cfg.col_t, "trip_id" if has_trip else None)

    cand_cols = [c for c in ["vehicle_jerk", "vehicle_accel", "vehicle_speed", "vehicle_odometer"] if c in df2.columns]
    if not cand_cols:
        raise ValueError(
            "No overwrite columns found in stage2_csv. Expected any of: "
            "vehicle_jerk, vehicle_accel, vehicle_speed, vehicle_odometer."
        )

    key = [cfg.col_vid, cfg.col_t]
    if has_trip:
        key = ["trip_id"] + key

    df2_key = df2[key + cand_cols].copy()
    dfm = df1.merge(df2_key, on=key, how="left", suffixes=("", "__s2"))

    s2_cols = {c: f"{c}__s2" for c in cand_cols}

    if cfg.replace_policy == "notna_all3":
        need = ["vehicle_accel", "vehicle_speed", "vehicle_odometer"]
        for c in need:
            if c not in cand_cols:
                raise ValueError("notna_all3 requires stage2 to include vehicle_accel/vehicle_speed/vehicle_odometer.")
        replace_mask = (
            dfm[s2_cols["vehicle_accel"]].notna()
            & dfm[s2_cols["vehicle_speed"]].notna()
            & dfm[s2_cols["vehicle_odometer"]].notna()
        )
    elif cfg.replace_policy == "notna_any":
        replace_mask = None
        for _, s2c in s2_cols.items():
            msk = dfm[s2c].notna()
            replace_mask = msk if replace_mask is None else (replace_mask | msk)
        replace_mask = replace_mask.fillna(False)
    else:
        raise ValueError("replace_policy must be one of: notna_any, notna_all3")

    for base_col, s2_col in s2_cols.items():
        if base_col in dfm.columns:
            dfm.loc[replace_mask, base_col] = dfm.loc[replace_mask, s2_col]
        else:
            dfm[base_col] = np.nan
            dfm.loc[replace_mask, base_col] = dfm.loc[replace_mask, s2_col]

    dfm.drop(columns=list(s2_cols.values()), inplace=True, errors="ignore")

    if "opt_source" in dfm.columns:
        dfm.drop(columns=["opt_source"], inplace=True, errors="ignore")

    stage1_has_any = (
        (dfm["vehicle_accel"].notna() if "vehicle_accel" in dfm.columns else False)
        | (dfm["vehicle_speed"].notna() if "vehicle_speed" in dfm.columns else False)
        | (dfm["vehicle_odometer"].notna() if "vehicle_odometer" in dfm.columns else False)
        | (dfm["vehicle_jerk"].notna() if "vehicle_jerk" in dfm.columns else False)
    )
    stage1_mask = (~replace_mask) & stage1_has_any

    dfm["opt_source"] = "none"
    dfm.loc[stage1_mask, "opt_source"] = "stage1"
    dfm.loc[replace_mask, "opt_source"] = "stage2"

    out_name = sanitize_filename(cfg.out_name)
    out_path = os.path.join(out_paths.merge_dir, f"{prefix_safe}_{out_name}.csv")
    dfm.to_csv(out_path, index=False, encoding=cfg.csv_encoding)

    logger.info(f"merged saved: {out_path}")
    logger.info(f"replaced rows: {int(replace_mask.sum()):,} / {len(dfm):,}")
    logger.info(f"time: {(time.time()-t0):.2f} sec")

    snap = {
        "cfg": asdict(cfg),
        "stats": {
            "rows": int(len(dfm)),
            "replaced": int(replace_mask.sum()),
            "policy": cfg.replace_policy,
            "stage2_cols": cand_cols,
            "key": key,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(out_paths.merge_dir, f"{prefix_safe}_merge_snapshot.json"), snap)

    return out_path


# =========================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("rvst_postprocess_v4.py (filter + merge)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp1 = sub.add_parser("filter", help="filter bad cases by jerk envelope (mid-points only)")
    sp1.add_argument("--stage1_csv", required=True)
    sp1.add_argument("--out_dir", required=True)
    sp1.add_argument("--prefix", default="fcd")
    sp1.add_argument("--jerk_env_csv", default="", help="Path to jerk envelope CSV (optional; will look under resources/ if empty)")
    sp1.add_argument("--accel_round_decimals", type=int, default=1)
    sp1.add_argument("--jerk_use_dt", action="store_true")
    sp1.add_argument("--dt", type=float, default=1.0)
    sp1.add_argument("--no_export_points", action="store_true")
    sp1.add_argument("--log_to_console", action="store_true", help="enable console logging (default: off)")

    sp2 = sub.add_parser("merge", help="merge stage2 result back to stage1")
    sp2.add_argument("--stage1_csv", required=True)
    sp2.add_argument("--stage2_csv", required=True)
    sp2.add_argument("--out_dir", required=True)
    sp2.add_argument("--prefix", default="fcd")
    sp2.add_argument("--replace_policy", choices=["notna_any", "notna_all3"], default="notna_any")
    sp2.add_argument("--log_to_console", action="store_true", help="enable console logging (default: off)")

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "filter":
        cfg = FilterConfig(
            stage1_csv=args.stage1_csv,
            out_dir=args.out_dir,
            prefix=args.prefix,
            jerk_env_csv=args.jerk_env_csv,
            accel_round_decimals=args.accel_round_decimals,
            jerk_use_dt=bool(args.jerk_use_dt),
            dt=float(args.dt),
            export_points=not bool(args.no_export_points),
            log_to_console=bool(args.log_to_console),
        )
        filter_bad_cases_by_jerk_envelope(cfg)

    elif args.cmd == "merge":
        cfg = MergeConfig(
            stage1_csv=args.stage1_csv,
            stage2_csv=args.stage2_csv,
            out_dir=args.out_dir,
            prefix=args.prefix,
            replace_policy=args.replace_policy,
            log_to_console=bool(args.log_to_console),
        )
        merge_stage1_stage2(cfg)

    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

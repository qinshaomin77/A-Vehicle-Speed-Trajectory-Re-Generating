# -*- coding: utf-8 -*-
"""
step25_neighbors_optimized.py

功能：
- 输入：FCD 标准化 CSV（至少包含 timestep_time, vehicle_id, vehicle_lane, vehicle_speed, vehicle_pos）
- 输入：lane连接关系 CSV（lane_id, next_lane_id）
- 输入：lane长度 CSV（lane_id, length）
- 输出：在原FCD基础上新增前后车信息字段后的 CSV/Parquet

核心优化：
- 并行粒度从 (timestep, lane) 改为 (timestep)：
  每个进程处理一个 timestep 的所有 lane，候选筛选仅在该 timestep 的小表 df_t 内进行
- 避免在 worker 内对全局大表反复过滤（你的原实现最大瓶颈）

说明：
- 保持你原始字段命名风格与 merge_asof 逻辑
- 仍支持向前/向后两层 lane 的拼接偏移（offset）构造
"""

from __future__ import annotations

import os
import re
import time
import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# -----------------------------
# 全局映射（供子进程使用）
# -----------------------------
G_NEXT_MAP: Dict[str, List[str]] = {}
G_PREV_MAP: Dict[str, List[str]] = {}
G_LEN_MAP: Dict[str, float] = {}


def _init_worker(next_map, prev_map, len_map):
    """多进程 initializer：把小字典放到每个子进程全局变量里（避免每次传参）"""
    global G_NEXT_MAP, G_PREV_MAP, G_LEN_MAP
    G_NEXT_MAP = next_map
    G_PREV_MAP = prev_map
    G_LEN_MAP = len_map


# -----------------------------
# 配置
# -----------------------------
@dataclass
class NeighborConfig:
    input_fcd_csv: str
    lane_next_csv: str
    lane_length_csv: str
    out_dir: str
    prefix: str = "FCD"

    # 并行
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

    # 输出格式
    out_format: str = "csv"   # "csv" or "parquet"
    csv_encoding: str = "utf-8-sig"

    # 连接层数：0=仅同车道；1=同车道+一层；2=同车道+两层
    hop: int = 2

    # 基础列名（与你现有一致）
    col_t: str = "timestep_time"
    col_id: str = "vehicle_id"
    col_lane: str = "vehicle_lane"
    col_v: str = "vehicle_speed"
    col_pos: str = "vehicle_pos"

    # 是否把 lane_length merge 回原表（通常会保留）
    merge_lane_length: bool = True


# -----------------------------
# 读取拓扑&长度映射
# -----------------------------
def build_lane_maps(lane_next_csv: str, lane_length_csv: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, float], pd.DataFrame]:
    lane_df = pd.read_csv(lane_next_csv)
    length_df = pd.read_csv(lane_length_csv)

    # 统一字段名
    if "lane_id" not in length_df.columns and "vehicle_lane" in length_df.columns:
        length_df = length_df.rename(columns={"vehicle_lane": "lane_id"})
    if "length" not in length_df.columns and "lane_length" in length_df.columns:
        length_df = length_df.rename(columns={"lane_length": "length"})

    length_df = length_df.rename(columns={"lane_id": "vehicle_lane", "length": "lane_length"})
    len_map = dict(zip(length_df["vehicle_lane"].astype(str), length_df["lane_length"].astype(float)))

    next_map: Dict[str, List[str]] = {}
    prev_map: Dict[str, List[str]] = {}
    for _, row in lane_df.iterrows():
        a = str(row["lane_id"])
        b = str(row["next_lane_id"])
        next_map.setdefault(a, []).append(b)
        prev_map.setdefault(b, []).append(a)

    return next_map, prev_map, len_map, length_df


def lane_to_edge(lane_id: str) -> str:
    s = str(lane_id).strip()
    if s.startswith(":"):
        return s
    return s.rsplit("_", 1)[0] if "_" in s else s


# -----------------------------
# 计算某 lane 的 next/prev 候选集合（含 offset）
# -----------------------------
def _collect_next_lanes_with_offset(lane_id: str, hop: int) -> Dict[str, float]:
    """
    返回 {lane: offset}，offset 表示从当前 lane 起点映射到该 lane 的累计偏移（向前）
    同 lane offset=0
    1-hop: next lane offset=当前 lane 长度
    2-hop: next2 lane offset=当前 lane 长度 + next1 lane 长度
    """
    out = {lane_id: 0.0}
    if hop <= 0:
        return out

    L0 = float(G_LEN_MAP.get(lane_id, 0.0))
    for n1 in G_NEXT_MAP.get(lane_id, []):
        out[n1] = min(out.get(n1, 1e18), L0)
        if hop >= 2:
            L1 = float(G_LEN_MAP.get(n1, 0.0))
            for n2 in G_NEXT_MAP.get(n1, []):
                out[n2] = min(out.get(n2, 1e18), L0 + L1)
    return out


def _collect_prev_lanes_with_offset(lane_id: str, hop: int) -> Dict[str, float]:
    """
    返回 {lane: offset}，offset 表示从当前 lane 起点回推到前序 lane 的累计长度（向后）
    同 lane offset=0
    1-hop: prev lane offset=prev lane 长度
    2-hop: prev2 lane offset=prev1 长度 + prev2 长度
    """
    out = {lane_id: 0.0}
    if hop <= 0:
        return out

    for p1 in G_PREV_MAP.get(lane_id, []):
        L1 = float(G_LEN_MAP.get(p1, 0.0))
        out[p1] = min(out.get(p1, 1e18), L1)
        if hop >= 2:
            for p2 in G_PREV_MAP.get(p1, []):
                L2 = float(G_LEN_MAP.get(p2, 0.0))
                out[p2] = min(out.get(p2, 1e18), L1 + L2)
    return out


# -----------------------------
# timestep 级处理（worker）
# -----------------------------
def process_one_timestep(args) -> pd.DataFrame:
    """
    输入：(t, df_t, hop, colnames...)
    返回：该 timestep 下每辆车的 前车/后车 匹配结果（行数=该 timestep 的车辆数）
    """
    (
        t,
        df_t,
        hop,
        col_t, col_id, col_lane, col_v, col_pos
    ) = args

    # 只保留必要列，减少排序/复制开销
    df_t = df_t[[col_t, col_id, col_lane, col_v, col_pos]].copy()
    df_t[col_lane] = df_t[col_lane].astype(str)

    # 结果收集（按 lane 合并后 concat）
    out_parts = []

    # 预先按 lane 分组（在该 timestep 小表内）
    for lane_id, df_lane in df_t.groupby(col_lane, sort=False):
        df_lane = df_lane.copy()
        df_lane["position"] = df_lane[col_pos].astype(float)
        df_lane = df_lane.sort_values("position")

        # ---------- next candidates ----------
        next_off = _collect_next_lanes_with_offset(lane_id, hop)
        next_lanes = list(next_off.keys())

        next_df = df_t[df_t[col_lane].isin(next_lanes)].copy()
        if not next_df.empty:
            next_df["position_preceding"] = next_df[col_pos].astype(float) + next_df[col_lane].map(next_off).fillna(0.0)
            next_df = next_df.sort_values("position_preceding").reset_index(drop=True)

        # ---------- prev candidates ----------
        prev_off = _collect_prev_lanes_with_offset(lane_id, hop)
        prev_lanes = list(prev_off.keys())

        prev_df = df_t[df_t[col_lane].isin(prev_lanes)].copy()
        if not prev_df.empty:
            prev_df["position_following"] = prev_df[col_pos].astype(float) - prev_df[col_lane].map(prev_off).fillna(0.0)
            prev_df = prev_df.sort_values("position_following").reset_index(drop=True)

        # merge_asof 要求 key 列升序
        # 前车匹配（direction='forward'）：找 position_preceding >= position 的最近值
        if next_df.empty:
            result_front = df_lane.copy()
            # 补齐列
            for c in [f"{col_id}_preceding", f"{col_pos}_preceding", f"{col_v}_preceding", f"{col_lane}_preceding", "position_preceding"]:
                result_front[c] = np.nan
        else:
            result_front = pd.merge_asof(
                df_lane,
                next_df,
                left_on="position",
                right_on="position_preceding",
                direction="forward",
                allow_exact_matches=False,
                suffixes=("", "_preceding"),
            )

        result_front["following_headway_distance"] = result_front["position_preceding"] - result_front["position"]

        # 后车匹配（direction='backward'）：找 position_following <= position 的最近值
        if prev_df.empty:
            result_both = result_front.copy()
            for c in [f"{col_id}_following", f"{col_pos}_following", f"{col_v}_following", f"{col_lane}_following", "position_following"]:
                result_both[c] = np.nan
        else:
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

        # 输出字段整理（保持你原来的命名习惯）
        result = result_both[[
            col_id, col_t,
            f"{col_id}_preceding", f"{col_pos}_preceding", f"{col_v}_preceding", f"{col_lane}_preceding",
            "following_headway_distance",
            f"{col_id}_following", f"{col_pos}_following", f"{col_v}_following", f"{col_lane}_following",
            "preceding_headway_distance",
        ]].rename(columns={
            f"{col_id}_preceding": "following_vehicle_id",
            f"{col_pos}_preceding": "following_flow_pos",
            f"{col_v}_preceding": "following_vehicle_speed",
            f"{col_lane}_preceding": "following_vehicle_lane",

            f"{col_id}_following": "preceding_vehicle_id",
            f"{col_pos}_following": "preceding_flow_pos",
            f"{col_v}_following": "preceding_vehicle_speed",
            f"{col_lane}_following": "preceding_vehicle_lane",
        })

        out_parts.append(result)

    if not out_parts:
        return pd.DataFrame(columns=[
            col_id, col_t,
            "following_vehicle_id", "following_flow_pos", "following_vehicle_speed", "following_vehicle_lane",
            "following_headway_distance",
            "preceding_vehicle_id", "preceding_flow_pos", "preceding_vehicle_speed", "preceding_vehicle_lane",
            "preceding_headway_distance",
        ])

    out_df = pd.concat(out_parts, ignore_index=True)
    return out_df


# -----------------------------
# 主入口：run()
# -----------------------------
def run(cfg: NeighborConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) 读取拓扑/长度映射
    next_map, prev_map, len_map, length_df = build_lane_maps(cfg.lane_next_csv, cfg.lane_length_csv)

    # 2) 读取FCD
    fcd_df = pd.read_csv(cfg.input_fcd_csv)
    # 基础列检查
    need = {cfg.col_t, cfg.col_id, cfg.col_lane, cfg.col_v, cfg.col_pos}
    miss = need - set(fcd_df.columns)
    if miss:
        raise ValueError(f"Input FCD is missing required columns: {sorted(miss)}")

    # 3) 可选 merge lane_length，并补 edge
    if cfg.merge_lane_length:
        fcd_df[cfg.col_lane] = fcd_df[cfg.col_lane].astype(str)
        length_df[cfg.col_lane] = length_df[cfg.col_lane].astype(str)
        fcd_df = pd.merge(fcd_df, length_df, on=cfg.col_lane, how="left")

    if "vehicle_edge" not in fcd_df.columns and cfg.col_lane in fcd_df.columns:
        fcd_df["vehicle_edge"] = fcd_df[cfg.col_lane].astype(str).apply(lane_to_edge)

    # 排序（保证 merge_asof 的稳定性）
    fcd_df = fcd_df.sort_values([cfg.col_id, cfg.col_t]).reset_index(drop=True)

    # 4) 只保留必要列给 worker（减少序列化数据量）
    fcd_small = fcd_df[[cfg.col_t, cfg.col_id, cfg.col_lane, cfg.col_v, cfg.col_pos]].copy()


    # 5) 构造 timestep 任务（按 timestep 分组）
    #    注意：这里的 df_t 会被 pickle 发送给子进程，数量比 (t,lane) 少很多
    tasks = []
    for t, df_t in fcd_small.groupby(cfg.col_t, sort=False):
        tasks.append((t, df_t, cfg.hop, cfg.col_t, cfg.col_id, cfg.col_lane, cfg.col_v, cfg.col_pos))

    # Determine worker count adaptively (to avoid "max_workers must be <= ...")
    cpu = os.cpu_count() or 1

    # Candidate strategy:
    # - If cfg.n_jobs > 0, try it first as a user override.
    # - Otherwise (or if it fails), try cpu//2, cpu//3, cpu//4, cpu//5
    # - If all fail, fall back to single-core.
    candidates = []
    if isinstance(cfg.n_jobs, int) and cfg.n_jobs > 0:
        candidates.append(cfg.n_jobs)
    for div in (2, 3, 4, 5):
        candidates.append(max(1, cpu // div))
    candidates.append(1)

    # de-duplicate while keeping order
    seen = set()
    cand_unique = []
    for n in candidates:
        if n not in seen:
            cand_unique.append(n)
            seen.add(n)

    print(f"[STEP01] timesteps: {len(tasks):,}  | cpu={cpu}  | hop={cfg.hop}")
    print(f"[STEP01] n_jobs candidates: {cand_unique}")

    # 6) Parallel processing (with adaptive fallback)
    t0 = time.time()
    results = []

    last_err = None
    for n_workers in cand_unique:
        try:
            if n_workers <= 1:
                # single-core fallback
                print("[STEP01] Running in single-core mode (n_jobs=1).")
                for task in tqdm(tasks, total=len(tasks), desc="🚗 Neighbor identification"):
                    results.append(process_one_timestep(task))
            else:
                print(f"[STEP01] Trying parallel mode with n_jobs={n_workers} ...")
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_worker,
                    initargs=(next_map, prev_map, len_map),
                ) as ex:
                    futs = [ex.submit(process_one_timestep, task) for task in tasks]
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="🚗 Neighbor identification"):
                        results.append(fut.result())
            # success
            cfg.n_jobs = n_workers  # record the final choice
            last_err = None
            break
        except ValueError as e:
            # Typical: "max_workers must be <= XX"
            last_err = e
            msg = str(e)
            if "max_workers" in msg:
                print(f"[STEP01] n_jobs={n_workers} is not allowed ({msg}). Falling back ...")
                results = []
                continue
            raise
        except Exception as e:
            # Other errors should not be silently retried
            raise

    if last_err is not None and not results:
        # If we exhausted candidates and still failed before producing any results
        raise last_err

    res_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    print(f"[STEP01] done, result rows={len(res_df):,}, time={(time.time()-t0)/60:.2f} min")

    # 7) merge 回原表
    merged = pd.merge(
        fcd_df,
        res_df,
        on=[cfg.col_id, cfg.col_t],
        how="left",
    ).sort_values([cfg.col_id, cfg.col_t]).reset_index(drop=True)

    # 8) 输出
    out_path = os.path.join(cfg.out_dir, f"{cfg.prefix}_neighbors.{cfg.out_format}")
    if cfg.out_format.lower() == "parquet":
        merged.to_parquet(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False, encoding=cfg.csv_encoding)

    print(f"✅ Output: {out_path}")
    return out_path


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Lane-based preceding/following identification (optimized)")
    p.add_argument("--input_fcd_csv", required=True)
    p.add_argument("--lane_next_csv", required=True, help="lane attributes CSV (lane_id,next_lane_id)")
    p.add_argument("--lane_length_csv", required=True, help="lane lengths CSV (lane_id,length)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="FCD150")

    p.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--hop", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--out_format", default="csv", choices=["csv", "parquet"])

    args = p.parse_args()

    cfg = NeighborConfig(
        input_fcd_csv=args.input_fcd_csv,
        lane_next_csv=args.lane_next_csv,
        lane_length_csv=args.lane_length_csv,
        out_dir=args.out_dir,
        prefix=args.prefix,
        n_jobs=args.n_jobs,
        hop=args.hop,
        out_format=args.out_format,
    )
    run(cfg)


if __name__ == "__main__":
    # Windows 多进程保护必须有
    main()

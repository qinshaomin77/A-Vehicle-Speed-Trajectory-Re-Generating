#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export lane connectivity and enumerate simple paths from a SUMO .net.xml file.

Inputs
------
Required arguments:
- --net: Path to a SUMO network file (*.net.xml). The script reads:
  * <connection from=... to=... fromLane=... toLane=... via=...>
  * <lane id=... length=...>

Optional arguments:
- --rou: Path to a SUMO route file (*.rou.xml). If provided, the script maps each (from_edge, to_edge)
  pair to a flow_id by selecting the <flow> with the largest 'vehsPerHour'. Expected element:
  * <flow id="..." from="edgeA" to="edgeB" vehsPerHour="...">

- --include-internal / --exclude-internal: Whether to keep internal lanes (IDs starting with ':').
  When excluding, internal 'via' connections are collapsed into direct lane->lane links.
- --max-steps: Maximum DFS steps to cap path enumeration.
- --excel: Write CSVs with UTF-8 BOM (utf-8-sig) for better Excel compatibility.
- --log-level: Logging verbosity (DEBUG, INFO, WARNING, ERROR).

Outputs
-------
All files are written to --output-dir.

1) {paths-stem}.csv — Path summary (one row per unique path at edge-lane granularity; consecutive duplicates merged)
   Columns:
   - path_id
   - path_len (meters)
   - n_edge_lanes
   - start_edge_lane
   - end_edge_lane
   - is_strict_start_edge (True if indegree==0 at edge level)
   - is_strict_end_edge   (True if outdegree==0 at edge level)
   - from_edge (first non-internal edge in the sequence)
   - to_edge   (last non-internal edge in the sequence)
   - flow_id   (if --rou provided; otherwise empty)

2) {paths-long-stem}.csv — Step-by-step path detail with cumulative length
   Columns:
   - path_id
   - EdgeLane_seq
   - EdgeLane_id
   - EdgeLane_length (meters)
   - cum_length (meters)
   - cum_length_prev (meters)
   - from_edge, to_edge, flow_id (same meaning as above)

3) {links-stem}.csv — Full lane topology table
   Columns:
   - lane_id
   - next_lane_id
   - lane_length (meters)
   - next_lane_length (meters)

4) {lengths-stem}.csv — Lane lengths table
   Columns:
   - lane_id
   - length (meters)

Notes
-----
- Default CSV encoding is UTF-8; pass --excel to write UTF-8 with BOM for Excel.
- Internal lanes appear in outputs only if --include-internal is used.

Notes
-----
- Path enumeration is DFS-based with a safety step limit.
- When excluding internal lanes, connections that use 'via' internal lanes are collapsed to direct lane->lane edges.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

import pandas as pd


# ----------------------------
# CLI & logging
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enumerate SUMO lane paths and export lane topology/lengths tables."
    )
    p.add_argument("--net", type=Path, required=True, help="Path to SUMO network file, e.g. net.net.xml")
    p.add_argument("--rou", type=Path, default=None, help="Optional path to rou.rou.xml to extract flow_id mapping.")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Directory to write outputs.")
    p.add_argument("--paths-stem", type=str, default="SUMO_NET_paths_edge",
                   help="File stem for paths summary CSV (without extension).")
    p.add_argument("--paths-long-stem", type=str, default="SUMO_NET_paths_long_edge",
                   help="File stem for paths long CSV (without extension).")
    p.add_argument("--links-stem", type=str, default="SUMO_NET_lane_attributes_edge",
                   help="File stem for full topology CSV (without extension).")
    p.add_argument("--lengths-stem", type=str, default="SUMO_NET_lane_lengths_edge",
                   help="File stem for lane lengths CSV (without extension).")
    p.add_argument("--include-internal", action="store_true", default=True,
                   help="Include internal lanes (IDs starting with ':'). Default: True.")
    p.add_argument("--exclude-internal", action="store_false", dest="include_internal",
                   help="Exclude internal lanes. Overrides --include-internal.")
    p.add_argument("--max-steps", type=int, default=100000,
                   help="Max DFS steps to avoid explosion. Default: 100000.")
    p.add_argument("--excel", action="store_true", help="Write CSV with utf-8-sig BOM (friendlier for Excel).")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return p.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ----------------------------
# Helpers
# ----------------------------
def is_internal_edge(x: str) -> bool:
    """Return True if the edge/lane id is an internal one (starts with ':')."""
    return isinstance(x, str) and x.startswith(":")


def first_non_internal(seq: Sequence[str]) -> Optional[str]:
    """Return the first non-internal edge in a sequence; None if not found."""
    for s in seq:
        if not is_internal_edge(s):
            return s
    return None


def last_non_internal(seq: Sequence[str]) -> Optional[str]:
    """Return the last non-internal edge in a sequence; None if not found."""
    for s in reversed(seq):
        if not is_internal_edge(s):
            return s
    return None


def lane_to_edge(s: str) -> str:
    """Map lane id (e.g., 'edge_0') to its edge id ('edge'). Keep internal lanes (':...') unchanged."""
    s = str(s)
    if s.startswith(":"):
        return s
    return s.rsplit("_", 1)[0] if "_" in s else s


def to_edge_lane_id(lane: str) -> str:
    """
    Normalize lane id for edge-lane sequence:
    - For non-internal lanes: drop the last underscore segment -> edge id
    - For internal lanes: keep as-is
    """
    s = str(lane)
    if s.startswith(":"):
        return s
    parts = s.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else s


def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in {name}: {miss}")


# ----------------------------
# rou.xml -> (from_edge, to_edge) -> flow_id (max vehsPerHour per pair)
# ----------------------------
def build_flow_map(rou_xml: Optional[Path]) -> Dict[Tuple[str, str], str]:
    flow_map: Dict[Tuple[str, str], str] = {}
    if rou_xml is None or not rou_xml.exists():
        logging.warning("rou.xml not provided or not found; flow_id will be missing.")
        return flow_map

    rtree = ET.parse(rou_xml)
    rroot = rtree.getroot()

    flows: List[Tuple[str, str, str, float]] = []
    for f in rroot.findall(".//flow"):
        fid = f.attrib.get("id")
        f_from = f.attrib.get("from")
        f_to = f.attrib.get("to")
        vph = f.attrib.get("vehsPerHour")
        try:
            vph_val = float(vph) if vph is not None else float("nan")
        except Exception:
            vph_val = float("nan")
        if fid and f_from and f_to:
            flows.append((fid, f_from, f_to, vph_val))

    df_flows = pd.DataFrame(flows, columns=["flow_id", "from_edge", "to_edge", "vehsPerHour"])
    if df_flows.empty:
        return flow_map

    # Keep the row with the largest vehsPerHour for each (from_edge, to_edge)
    df_flows = (
        df_flows.sort_values(["from_edge", "to_edge", "vehsPerHour"], ascending=[True, True, False])
        .drop_duplicates(subset=["from_edge", "to_edge"], keep="first")
    )
    flow_map = {(a, b): fid for fid, a, b in df_flows[["flow_id", "from_edge", "to_edge"]].itertuples(index=False, name=None)}
    return flow_map


def lookup_flow(flow_map: Dict[Tuple[str, str], str], from_edge: Optional[str], to_edge: Optional[str]) -> Optional[str]:
    if not from_edge or not to_edge:
        return None
    return flow_map.get((from_edge, to_edge))


# ----------------------------
# XML parsing -> links and lane lengths
# ----------------------------
def parse_net_xml(
    net_xml: Path,
    include_internal: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse SUMO .net.xml and build:
      - df_links: DataFrame[lane_id, next_lane_id]
      - df_lane_len: DataFrame[lane_id, length]
    When excluding internal lanes, 'via' connections are collapsed (from_lane -> to_lane).
    """
    tree = ET.parse(net_xml)
    root = tree.getroot()

    lane_links: List[Tuple[str, Optional[str]]] = []
    # Build lane -> next_lane links
    for conn in root.findall(".//connection"):
        from_edge = conn.attrib.get("from")
        to_edge = conn.attrib.get("to")
        from_lane = conn.attrib.get("fromLane")
        to_lane = conn.attrib.get("toLane")
        via = conn.attrib.get("via")

        if from_edge and to_edge and from_lane is not None and to_lane is not None:
            from_lane_id = f"{from_edge}_{from_lane}"
            to_lane_id = f"{to_edge}_{to_lane}"

            if via:
                # If including internal lanes, keep the via lane as separate step
                if include_internal:
                    lane_links.append((from_lane_id, via))
                    lane_links.append((via, to_lane_id))
                else:
                    # Collapse internal via to a direct connection
                    lane_links.append((from_lane_id, to_lane_id))
            else:
                lane_links.append((from_lane_id, to_lane_id))

    df_links = pd.DataFrame(lane_links, columns=["lane_id", "next_lane_id"]).drop_duplicates()

    # Lane lengths
    lane_info: List[Tuple[str, float]] = []
    for lane in root.findall(".//lane"):
        lane_id = lane.attrib.get("id")
        length = lane.attrib.get("length")
        if lane_id and length:
            lane_info.append((lane_id, float(length)))
    df_lane_len = pd.DataFrame(lane_info, columns=["lane_id", "length"])

    # Optionally drop internal lanes entirely
    if not include_internal:
        df_lane_len = df_lane_len[~df_lane_len["lane_id"].map(is_internal_edge)].reset_index(drop=True)
        # Remove links that point to or from internal lanes
        df_links = df_links[
            ~df_links["lane_id"].map(is_internal_edge) & ~df_links["next_lane_id"].fillna("").map(is_internal_edge)
        ].reset_index(drop=True)

    # Ensure lanes with no outgoing link are kept (next=None)
    unused = df_lane_len[~df_lane_len["lane_id"].isin(df_links["lane_id"])]
    if not unused.empty:
        df_links = pd.concat(
            [df_links, pd.DataFrame({"lane_id": unused["lane_id"].values, "next_lane_id": [None] * len(unused)})],
            ignore_index=True,
        )

    return df_links, df_lane_len


# ----------------------------
# Graph + DFS path enumeration
# ----------------------------
def build_graph(df_links: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str], List[str], Dict[str, int], Dict[str, int]]:
    """Build adjacency and compute indegree/outdegree plus start/end nodes."""
    adj: Dict[str, List[str]] = defaultdict(list)
    indeg_tmp: Dict[str, int] = defaultdict(int)
    nodes = set(df_links["lane_id"].tolist()) | set(df_links["next_lane_id"].dropna().tolist())

    for a, b in df_links.itertuples(index=False):
        if b is not None:
            adj[a].append(b)
            indeg_tmp[b] += 1
        nodes.add(a)
        if b is not None:
            nodes.add(b)

    # Deterministic order for debugging/reproducibility
    for k in list(adj.keys()):
        adj[k] = sorted(adj[k])

    outdeg = {u: len(adj.get(u, [])) for u in nodes}
    indeg = {u: indeg_tmp.get(u, 0) for u in nodes}

    starts = [u for u in nodes if indeg.get(u, 0) == 0]
    if not starts:
        # Fallback: pick nodes where indeg/outdeg is not both 1 (likely branching/terminal points)
        seeds = [u for u in nodes if not (indeg.get(u, 0) == 1 and outdeg.get(u, 0) == 1)]
        starts = seeds if seeds else list(nodes)

    sinks = [u for u in nodes if outdeg.get(u, 0) == 0]
    return adj, starts, sinks, indeg, outdeg


def enumerate_paths(
    adj: Dict[str, List[str]],
    starts: List[str],
    max_steps: int,
) -> List[List[str]]:
    """Enumerate simple paths using iterative DFS with a step cap."""
    all_paths: List[List[str]] = []
    step_counter = 0

    for s in starts:
        stack: List[List[str]] = [[s]]
        while stack:
            path = stack.pop()
            step_counter += 1
            if step_counter > max_steps:
                logging.warning("Search steps exceed max limit; early stop.")
                stack.clear()
                break

            u = path[-1]
            succs = adj.get(u, [])

            if len(succs) == 0:
                # Reached a sink
                all_paths.append(path)
                continue

            for v in succs:
                if v in path:
                    # Avoid cycles within current path; record the partial path
                    all_paths.append(path.copy())
                    continue
                stack.append(path + [v])

    # Deduplicate identical paths
    seen = set()
    unique_paths: List[List[str]] = []
    for p in all_paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_paths.append(p)
    return unique_paths


# ----------------------------
# Path tables (compact & long)
# ----------------------------
def build_path_tables(
    all_paths: List[List[str]],
    df_lane_len: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From raw lane-level paths, build:
      - df_paths_long_unique: compacted edge-lane steps with cumulative length
      - df_paths: summary per unique path
    """
    lane_len_map = dict(zip(df_lane_len["lane_id"], df_lane_len["length"]))

    # Raw long table over lanes; then normalize to edge-level ids ("EdgeLane_id")
    rows: List[Dict] = []
    for i, p in enumerate(all_paths):
        cum = 0.0
        for k, lane in enumerate(p):
            L = lane_len_map.get(lane, float("nan"))
            cum += (L if pd.notna(L) else 0.0)
            rows.append(
                {
                    "path_id": i,
                    "EdgeLane_seq": k,
                    "EdgeLane_id": to_edge_lane_id(lane),
                    "EdgeLane_length": L,
                    "cum_length_raw": cum,  # kept for debugging; recomputed later
                }
            )
    df_paths_long = (
        pd.DataFrame(rows).sort_values(["path_id", "EdgeLane_seq"]).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["path_id", "EdgeLane_seq", "EdgeLane_id", "EdgeLane_length", "cum_length_raw"])
    )

    if df_paths_long.empty:
        return pd.DataFrame(columns=["path_id", "path_len", "n_edge_lanes", "start_edge_lane", "end_edge_lane"]), df_paths_long

    # Collapse consecutive duplicates of the same EdgeLane_id within a path (sum lengths)
    blk = df_paths_long.copy()
    blk["__chg__"] = (blk["EdgeLane_id"] != blk.groupby("path_id")["EdgeLane_id"].shift()).astype(int)
    blk["__block__"] = blk.groupby("path_id")["__chg__"].cumsum()

    df_compact = (
        blk.groupby(["path_id", "__block__", "EdgeLane_id"], as_index=False)
        .agg(EdgeLane_length=("EdgeLane_length", "sum"))
        .sort_values(["path_id", "__block__"])
        .reset_index(drop=True)
    )

    # Deduplicate whole paths across different starts: signature is the tuple of EdgeLane_id
    sig = df_compact.groupby("path_id")["EdgeLane_id"].apply(tuple).reset_index(name="signature")
    sig["new_path_id"] = pd.factorize(sig["signature"])[0]

    # Map old path_id -> new unified id
    old2new = dict(zip(sig["path_id"], sig["new_path_id"]))
    df_compact["path_id"] = df_compact["path_id"].map(old2new)

    # Reorder inside each new path and recompute cumulative length
    df_paths_long_unique = (
        df_compact.sort_values(["path_id", "__block__"])
        .drop(columns=["__block__", "__chg__"], errors="ignore")
        .reset_index(drop=True)
    )
    df_paths_long_unique["EdgeLane_seq"] = df_paths_long_unique.groupby("path_id").cumcount()
    df_paths_long_unique["cum_length"] = df_paths_long_unique.groupby("path_id", sort=False)["EdgeLane_length"].cumsum()
    df_paths_long_unique["cum_length_prev"] = (
        df_paths_long_unique.groupby("path_id", sort=False)["cum_length"].shift(1).fillna(0.0)
    )

    # Summary table per path
    df_paths = (
        df_paths_long_unique.groupby("path_id")
        .agg(
            path_len=("cum_length", "max"),
            n_edge_lanes=("EdgeLane_id", "count"),
            start_edge_lane=("EdgeLane_id", "first"),
            end_edge_lane=("EdgeLane_id", "last"),
        )
        .reset_index()
    )

    return df_paths, df_paths_long_unique


def strict_start_end_flags(df_links: pd.DataFrame, include_internal: bool) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build edge-level indegree/outdegree from lane links and return dictionaries for start/end flagging.
    """
    df_e_edges = (
        df_links.dropna(subset=["next_lane_id"])
        .assign(edge_id=lambda d: d["lane_id"].map(lane_to_edge),
                next_edge_id=lambda d: d["next_lane_id"].map(lane_to_edge))
        .loc[lambda d: d["edge_id"] != d["next_edge_id"], ["edge_id", "next_edge_id"]]
        .drop_duplicates()
    )

    if not include_internal:
        df_e_edges = df_e_edges[
            ~df_e_edges["edge_id"].map(is_internal_edge) & ~df_e_edges["next_edge_id"].map(is_internal_edge)
        ]

    outdeg_e = Counter(df_e_edges["edge_id"])
    indeg_e = Counter(df_e_edges["next_edge_id"])
    edge_nodes = set(df_e_edges["edge_id"]).union(set(df_e_edges["next_edge_id"]))
    outdeg_e = {u: outdeg_e.get(u, 0) for u in edge_nodes}
    indeg_e = {u: indeg_e.get(u, 0) for u in edge_nodes}
    return indeg_e, outdeg_e


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("sumo_paths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enc = "utf-8-sig" if args.excel else "utf-8"

    # 1) Load net.xml -> links + lane lengths
    log.info("Parsing net XML: %s", args.net)
    df_links, df_lane_len = parse_net_xml(args.net, include_internal=args.include_internal)

    log.info("Links: %d rows, Lane lengths: %d rows", len(df_links), len(df_lane_len))

    # 2) Build graph + degrees + start/sink nodes
    adj, starts, sinks, indeg, outdeg = build_graph(df_links)
    log.info("Nodes: %d, Edges: %d, Starts: %d, Sinks: %d",
             len(indeg), sum(len(v) for v in adj.values()), len(starts), len(sinks))

    # 3) Enumerate simple paths (DFS with cap)
    log.info("Enumerating paths (max steps=%d)...", args.max_steps)
    all_paths = enumerate_paths(adj, starts, max_steps=args.max_steps)
    log.info("Raw path count: %d", len(all_paths))

    # 4) Build compact & long path tables
    df_paths, df_paths_long_unique = build_path_tables(all_paths, df_lane_len)
    log.info("Unique paths after compaction: %d", len(df_paths))

    # 5) Strict start/end flags at edge level
    indeg_e, outdeg_e = strict_start_end_flags(df_links, include_internal=args.include_internal)
    df_paths["is_strict_start_edge"] = df_paths["start_edge_lane"].map(lambda x: indeg_e.get(x, 0) == 0)
    df_paths["is_strict_end_edge"] = df_paths["end_edge_lane"].map(lambda x: outdeg_e.get(x, 0) == 0)

    # 6) Optional: flow_id via rou.xml (per path by from_edge/to_edge)
    flow_map = build_flow_map(args.rou)

    # Compute from_edge/to_edge/flow_id using the compact paths table
    g = df_paths_long_unique.groupby("path_id", sort=False)["EdgeLane_id"].apply(list).reset_index(name="edge_seq")
    g["from_edge"] = g["edge_seq"].apply(first_non_internal)
    g["to_edge"] = g["edge_seq"].apply(last_non_internal)
    g["flow_id"] = g.apply(lambda r: lookup_flow(flow_map, r["from_edge"], r["to_edge"]), axis=1)

    # Merge these to both summary and long tables
    df_paths = df_paths.merge(g[["path_id", "from_edge", "to_edge", "flow_id"]], on="path_id", how="left")
    df_paths_long = df_paths_long_unique.sort_values(["path_id", "EdgeLane_seq"]).reset_index(drop=True)
    df_paths_long = df_paths_long.merge(g[["path_id", "from_edge", "to_edge", "flow_id"]], on="path_id", how="left")

    # 7) Build "full topology" table (cover all lanes, with lengths of current and next)
    all_lanes = pd.Series(df_lane_len["lane_id"].unique(), name="lane_id")
    missing_as_from = all_lanes[~all_lanes.isin(df_links["lane_id"])]
    df_missing = pd.DataFrame({"lane_id": missing_as_from, "next_lane_id": None})

    df_links_full = (
        pd.concat([df_links[["lane_id", "next_lane_id"]], df_missing], ignore_index=True)
        .drop_duplicates()
        .merge(df_lane_len.rename(columns={"length": "lane_length"}), on="lane_id", how="left")
        .merge(
            df_lane_len.rename(columns={"lane_id": "next_lane_id", "length": "next_lane_length"}),
            on="next_lane_id",
            how="left",
        )
        [["lane_id", "next_lane_id", "lane_length", "next_lane_length"]]
        .sort_values(by=["lane_id", "next_lane_id"], na_position="last")
        .reset_index(drop=True)
    )

    # 8) Lane lengths (deduplicated)
    df_lane_length_all = df_lane_len.drop_duplicates(subset=["lane_id"]).sort_values("lane_id").reset_index(drop=True)

    # 9) Save outputs
    paths_path = args.output_dir / f"{args.paths_stem}.csv"
    paths_long_path = args.output_dir / f"{args.paths_long_stem}.csv"
    links_full_path = args.output_dir / f"{args.links_stem}.csv"
    lane_length_path = args.output_dir / f"{args.lengths_stem}.csv"

    df_paths.to_csv(paths_path, index=False, encoding=enc)
    df_paths_long.to_csv(paths_long_path, index=False, encoding=enc)
    df_links_full.to_csv(links_full_path, index=False, encoding="utf-8")  # topology usually fine w/o BOM
    df_lane_length_all.to_csv(lane_length_path, index=False, encoding="utf-8")

    logging.info("Saved paths summary to: %s", paths_path)
    logging.info("Saved paths long to: %s", paths_long_path)
    logging.info("Saved lane topology to: %s", links_full_path)
    logging.info("Saved lane lengths to: %s", lane_length_path)

    # Final peek (head) for quick sanity
    logging.debug("Paths summary head:\n%s", df_paths.head())
    logging.debug("Paths long head:\n%s", df_paths_long.head())


if __name__ == "__main__":
    main()

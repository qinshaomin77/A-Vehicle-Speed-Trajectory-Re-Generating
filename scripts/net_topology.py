# -*- coding: utf-8 -*-
"""
rvst_core/net_topology.py

Input:
    - SUMO net.xml
    - (optional) rou.xml (for flow_id mapping)
Output (DataFrames + optional CSV exports):
    - df_paths: path-level summary (edge-lane compact)
    - df_paths_long: segment-level detail for each path
    - df_lane_links_full: lane topology table (lane_id -> next_lane_id) with lengths
    - df_lane_lengths: lane length table

Design goals:
    - no hard-coded paths
    - callable from pipeline/UI
    - robust to missing rou.xml, missing lengths, dangling lanes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import xml.etree.ElementTree as ET
import pandas as pd


# --------------------------- Config ---------------------------

@dataclass(frozen=True)
class NetTopologyConfig:
    net_xml: Path
    out_dir: Optional[Path] = None

    rou_xml: Optional[Path] = None
    include_internal: bool = True
    max_steps: int = 100_000
    debug: bool = False

    export_paths: bool = True
    export_lane_tables: bool = True
    encoding: str = "utf-8-sig"


# --------------------------- Core ---------------------------

class NetTopologyExtractor:
    def __init__(self, cfg: NetTopologyConfig):
        self.cfg = cfg
        self.cfg_net = NetTopologyConfig(
            net_xml=Path(cfg.net_xml),
            out_dir=Path(cfg.out_dir) if cfg.out_dir is not None else None,
            rou_xml=Path(cfg.rou_xml) if cfg.rou_xml is not None else None,
            include_internal=cfg.include_internal,
            max_steps=cfg.max_steps,
            debug=cfg.debug,
            export_paths=cfg.export_paths,
            export_lane_tables=cfg.export_lane_tables,
            encoding=cfg.encoding,
        )

    # ---------- helpers ----------
    @staticmethod
    def _is_internal(s: Optional[str]) -> bool:
        return isinstance(s, str) and s.startswith(":")

    @staticmethod
    def _lane_to_edge_id(lane_id: str) -> str:
        """SUMO lane_id: edgeId_index ; internal lanes begin with ':' and are kept."""
        s = str(lane_id).strip()
        if s.startswith(":"):
            return s
        return s.rsplit("_", 1)[0] if "_" in s else s

    @staticmethod
    def _first_non_internal(seq: List[str]) -> Optional[str]:
        for x in seq:
            if not NetTopologyExtractor._is_internal(x):
                return x
        return None

    @staticmethod
    def _last_non_internal(seq: List[str]) -> Optional[str]:
        for x in reversed(seq):
            if not NetTopologyExtractor._is_internal(x):
                return x
        return None

    # ---------- parse rou.xml flows ----------
    def _parse_flow_map(self) -> Dict[Tuple[str, str], str]:
        """Return map: (from_edge, to_edge) -> flow_id with max vehsPerHour per pair."""
        rou = self.cfg_net.rou_xml
        if rou is None or not rou.exists():
            if self.cfg_net.debug:
                print(f"[INFO] rou.xml not provided or not exists: {rou}. flow_id will be None.")
            return {}

        tree = ET.parse(str(rou))
        root = tree.getroot()

        flows = []
        for f in root.findall(".//flow"):
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

        if not flows:
            return {}

        df = pd.DataFrame(flows, columns=["flow_id", "from_edge", "to_edge", "vehsPerHour"])
        df = (df.sort_values(["from_edge", "to_edge", "vehsPerHour"], ascending=[True, True, False])
                .drop_duplicates(subset=["from_edge", "to_edge"], keep="first"))

        return {(a, b): fid for fid, a, b in df[["flow_id", "from_edge", "to_edge"]].itertuples(index=False, name=None)}

    # ---------- parse net.xml (lane links & lane lengths) ----------
    def _parse_net(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        net = self.cfg_net.net_xml
        if not net.exists():
            raise FileNotFoundError(f"net.xml not found: {net}")

        tree = ET.parse(str(net))
        root = tree.getroot()

        # lane connection graph (lane_id -> next_lane_id)
        lane_links: List[Tuple[str, Optional[str]]] = []
        for conn in root.findall(".//connection"):
            from_edge = conn.attrib.get("from")
            to_edge = conn.attrib.get("to")
            from_lane = conn.attrib.get("fromLane")
            to_lane = conn.attrib.get("toLane")
            via = conn.attrib.get("via")

            if not (from_edge and to_edge and from_lane is not None and to_lane is not None):
                continue

            from_lane_id = f"{from_edge}_{from_lane}"
            to_lane_id = f"{to_edge}_{to_lane}"

            # if include_internal=False, skip internal intermediate edges/lane if they appear
            if (not self.cfg_net.include_internal) and (self._is_internal(from_lane_id) or self._is_internal(to_lane_id) or self._is_internal(via)):
                continue

            if via:
                # via is a lane id (internal lane)
                if self.cfg_net.include_internal or (not self._is_internal(via)):
                    lane_links.append((from_lane_id, via))
                    lane_links.append((via, to_lane_id))
            else:
                lane_links.append((from_lane_id, to_lane_id))

        df_links = pd.DataFrame(lane_links, columns=["lane_id", "next_lane_id"]).drop_duplicates()

        # lane lengths
        lane_info = []
        for lane in root.findall(".//lane"):
            lane_id = lane.attrib.get("id")
            length = lane.attrib.get("length")
            if lane_id is None or length is None:
                continue
            if (not self.cfg_net.include_internal) and self._is_internal(lane_id):
                continue
            try:
                lane_info.append((lane_id, float(length)))
            except Exception:
                lane_info.append((lane_id, float("nan")))
        df_lane_len = pd.DataFrame(lane_info, columns=["lane_id", "length"]).drop_duplicates()

        # ensure every lane exists as a "from" node at least once (next=None)
        if not df_lane_len.empty:
            unused = df_lane_len[~df_lane_len["lane_id"].isin(df_links["lane_id"])]
            if not unused.empty:
                df_links = pd.concat(
                    [df_links, pd.DataFrame({"lane_id": unused["lane_id"].values, "next_lane_id": [None] * len(unused)})],
                    ignore_index=True
                ).drop_duplicates()

        return df_links, df_lane_len

    # ---------- build adjacency & degrees ----------
    def _build_adj(self, df_links: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, int]]:
        adj = defaultdict(list)
        indeg_tmp = defaultdict(int)

        nodes = set(df_links["lane_id"].tolist()) | set(df_links["next_lane_id"].dropna().tolist())
        for a, b in df_links.itertuples(index=False):
            if b is not None:
                adj[a].append(b)
                indeg_tmp[b] += 1

        # stable ordering
        for k in list(adj.keys()):
            adj[k] = sorted(adj[k])

        outdeg = {u: len(adj.get(u, [])) for u in nodes}
        indeg = {u: indeg_tmp.get(u, 0) for u in nodes}
        return dict(adj), indeg, outdeg

    # ---------- enumerate paths (iterative DFS, avoid cycles) ----------
    def _enumerate_paths(self, adj: Dict[str, List[str]], indeg: Dict[str, int], outdeg: Dict[str, int]) -> List[List[str]]:
        nodes = set(indeg.keys()) | set(outdeg.keys()) | set(adj.keys())

        starts = [u for u in nodes if indeg.get(u, 0) == 0]
        if not starts:
            # fallback seeds: nodes that are not pure 1-in-1-out
            seeds = [u for u in nodes if not (indeg.get(u, 0) == 1 and outdeg.get(u, 0) == 1)]
            starts = seeds if seeds else list(nodes)

        if self.cfg_net.debug:
            sinks = [u for u in nodes if outdeg.get(u, 0) == 0]
            print(f"[DEBUG] nodes={len(nodes)} starts={len(starts)} sinks={len(sinks)} edges={sum(len(v) for v in adj.values())}")

        all_paths: List[List[str]] = []
        step_counter = 0

        for s in starts:
            stack = [[s]]
            while stack:
                path = stack.pop()
                step_counter += 1
                if step_counter > self.cfg_net.max_steps:
                    print("[WARN] Path enumeration exceeded max_steps, stopped early.")
                    return self._unique_paths(all_paths)

                u = path[-1]
                succs = adj.get(u, [])

                if not succs:
                    all_paths.append(path)
                    continue

                for v in succs:
                    if v in path:
                        # cycle avoidance: record current prefix path
                        all_paths.append(path.copy())
                        continue
                    stack.append(path + [v])

        return self._unique_paths(all_paths)

    @staticmethod
    def _unique_paths(paths: List[List[str]]) -> List[List[str]]:
        seen = set()
        out = []
        for p in paths:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                out.append(p)
        return out

    # ---------- compact to EdgeLane id & cross-path dedup ----------
    def _build_paths_tables(
        self,
        all_paths: List[List[str]],
        lane_len_map: Dict[str, float],
        flow_map: Dict[Tuple[str, str], str],
        df_links: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # long (lane path) -> long (edge-lane path)
        rows = []
        for pid, p in enumerate(all_paths):
            cum = 0.0
            for k, lane in enumerate(p):
                L = float(lane_len_map.get(lane, float("nan")))
                cum += (L if pd.notna(L) else 0.0)
                edge_lane_id = self._lane_to_edge_id(lane)
                rows.append(
                    dict(
                        path_id=pid,
                        EdgeLane_seq=k,
                        EdgeLane_id=edge_lane_id,
                        EdgeLane_length=L,
                        cum_length_raw=cum,
                    )
                )
        df_long = pd.DataFrame(rows).sort_values(["path_id", "EdgeLane_seq"]).reset_index(drop=True)

        if df_long.empty:
            df_paths = pd.DataFrame(columns=["path_id", "path_len", "n_edge_lanes", "start_edge_lane", "end_edge_lane"])
            return df_paths, df_long, pd.DataFrame(columns=["path_id", "edge_seq", "from_edge", "to_edge", "flow_id"])

        # compact consecutive identical EdgeLane_id inside each path
        blk = df_long.copy()
        blk["__chg__"] = (blk["EdgeLane_id"] != blk.groupby("path_id")["EdgeLane_id"].shift()).astype(int)
        blk["__block__"] = blk.groupby("path_id")["__chg__"].cumsum()

        df_compact = (blk.groupby(["path_id", "__block__", "EdgeLane_id"], as_index=False)
                        .agg(EdgeLane_length=("EdgeLane_length", "sum"))
                        .sort_values(["path_id", "__block__"])
                        .reset_index(drop=True))

        # cross-path dedup by signature
        sig = (df_compact.groupby("path_id")["EdgeLane_id"].apply(tuple).reset_index(name="signature"))
        sig["new_path_id"] = pd.factorize(sig["signature"])[0]

        rep = (sig.sort_values("path_id")
                 .drop_duplicates("new_path_id", keep="first")
                 .rename(columns={"path_id": "rep_old_path_id"})[["new_path_id", "rep_old_path_id"]])

        old2new = dict(zip(sig["path_id"], sig["new_path_id"]))
        df_compact["new_path_id"] = df_compact["path_id"].map(old2new)

        keep_old_ids = set(rep["rep_old_path_id"].tolist())
        df_unique = df_compact[df_compact["path_id"].isin(keep_old_ids)].copy()

        # reassign path_id + recompute sequence and cumulative lengths
        df_unique["path_id"] = df_unique["new_path_id"]
        df_unique = df_unique.sort_values(["path_id", "__block__"]).drop(columns=["new_path_id"], errors="ignore")
        df_unique["EdgeLane_seq"] = df_unique.groupby("path_id").cumcount()
        df_unique["cum_length"] = df_unique.groupby("path_id")["EdgeLane_length"].cumsum()
        df_unique["cum_length_prev"] = df_unique.groupby("path_id")["cum_length"].shift(1).fillna(0.0)

        # summary table
        df_paths = (df_unique.groupby("path_id").agg(
            path_len=("cum_length", "max"),
            n_edge_lanes=("EdgeLane_id", "count"),
            start_edge_lane=("EdgeLane_id", "first"),
            end_edge_lane=("EdgeLane_id", "last"),
        ).reset_index())

        # strict start/end on edge-level graph (collapse df_links to edge)
        df_e = (df_links.dropna(subset=["next_lane_id"])
                    .assign(edge_id=lambda d: d["lane_id"].map(self._lane_to_edge_id),
                            next_edge_id=lambda d: d["next_lane_id"].map(self._lane_to_edge_id))
                    .loc[lambda d: d["edge_id"] != d["next_edge_id"], ["edge_id", "next_edge_id"]]
                    .drop_duplicates())

        outdeg_e = Counter(df_e["edge_id"])
        indeg_e = Counter(df_e["next_edge_id"])
        edge_nodes = set(df_e["edge_id"]).union(set(df_e["next_edge_id"]))

        df_paths["is_strict_start_edge"] = df_paths["start_edge_lane"].map(lambda x: indeg_e.get(x, 0) == 0)
        df_paths["is_strict_end_edge"] = df_paths["end_edge_lane"].map(lambda x: outdeg_e.get(x, 0) == 0)

        # attach from/to/flow_id for each path_id
        g = df_unique.groupby("path_id", sort=False)["EdgeLane_id"].apply(list).reset_index(name="edge_seq")
        g["from_edge"] = g["edge_seq"].apply(self._first_non_internal)
        g["to_edge"] = g["edge_seq"].apply(self._last_non_internal)
        if flow_map:
            g["flow_id"] = g.apply(lambda r: flow_map.get((r["from_edge"], r["to_edge"])), axis=1)
        else:
            g["flow_id"] = None

        df_paths_long = df_unique.merge(g[["path_id", "flow_id", "from_edge", "to_edge"]], on="path_id", how="left")
        return df_paths, df_paths_long, g

    # ---------- lane tables (full coverage) ----------
    def _build_lane_tables(self, df_links: pd.DataFrame, df_lane_len: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # ensure all lanes appear
        all_lanes = pd.Series(df_lane_len["lane_id"].unique(), name="lane_id")
        missing_as_from = all_lanes[~all_lanes.isin(df_links["lane_id"])]
        df_missing = pd.DataFrame({"lane_id": missing_as_from, "next_lane_id": None})

        df_links_full = pd.concat([df_links[["lane_id", "next_lane_id"]], df_missing], ignore_index=True).drop_duplicates()

        df_links_full = df_links_full.merge(
            df_lane_len.rename(columns={"length": "lane_length"}),
            on="lane_id", how="left"
        ).merge(
            df_lane_len.rename(columns={"lane_id": "next_lane_id", "length": "next_lane_length"}),
            on="next_lane_id", how="left"
        )

        df_links_full = df_links_full[["lane_id", "next_lane_id", "lane_length", "next_lane_length"]] \
                                     .sort_values(["lane_id", "next_lane_id"], na_position="last") \
                                     .reset_index(drop=True)

        df_lane_lengths = df_lane_len.drop_duplicates("lane_id").sort_values("lane_id").reset_index(drop=True)
        return df_links_full, df_lane_lengths

    # ---------- public API ----------
    def run(self) -> Dict[str, pd.DataFrame]:
        flow_map = self._parse_flow_map()
        df_links, df_lane_len = self._parse_net()

        adj, indeg, outdeg = self._build_adj(df_links)
        all_paths = self._enumerate_paths(adj, indeg, outdeg)

        lane_len_map = dict(zip(df_lane_len["lane_id"], df_lane_len["length"]))
        df_paths, df_paths_long, _ = self._build_paths_tables(all_paths, lane_len_map, flow_map, df_links)

        df_links_full, df_lane_lengths = self._build_lane_tables(df_links, df_lane_len)

        # optional exports
        if self.cfg_net.out_dir is not None:
            out_dir = self.cfg_net.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.cfg_net.export_paths:
                df_paths.to_csv(out_dir / "SUMO_NET_paths_edge.csv", index=False, encoding=self.cfg_net.encoding)
                df_paths_long.to_csv(out_dir / "SUMO_NET_paths_long_edge.csv", index=False, encoding=self.cfg_net.encoding)

            if self.cfg_net.export_lane_tables:
                df_links_full.to_csv(out_dir / "SUMO_NET_lane_attributes_edge.csv", index=False, encoding=self.cfg_net.encoding)
                df_lane_lengths.to_csv(out_dir / "SUMO_NET_lane_lengths_edge.csv", index=False, encoding=self.cfg_net.encoding)

        if self.cfg_net.debug:
            print("[DEBUG] df_paths:", df_paths.shape, " df_paths_long:", df_paths_long.shape)
            print("[DEBUG] df_links_full:", df_links_full.shape, " df_lane_lengths:", df_lane_lengths.shape)

        return {
            "df_paths": df_paths,
            "df_paths_long": df_paths_long,
            "df_lane_links_full": df_links_full,
            "df_lane_lengths": df_lane_lengths,
        }


# --------------------------- Example usage (CLI-like) ---------------------------
if __name__ == "__main__":
    cfg = NetTopologyConfig(
        net_xml=Path("path/to/net.xml"),
        rou_xml=None,                 # optional
        out_dir=Path("output_dir"),   # optional
        include_internal=True,
        max_steps=100_000,
        debug=True,
        export_paths=True,
        export_lane_tables=True,
    )
    extractor = NetTopologyExtractor(cfg)
    out = extractor.run()
    print(out["df_paths"].head())
    print(out["df_paths_long"].head())

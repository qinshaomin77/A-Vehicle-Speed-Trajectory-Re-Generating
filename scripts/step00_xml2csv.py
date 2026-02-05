# -*- coding: utf-8 -*-
r"""
00xml转csv.py  (optimized)

功能：
1) 调用外部 xml2csv.py 工具：FCD.xml -> 分号分隔 CSV（out_base.csv）
2) 将分号 CSV 转为标准逗号 CSV（可选）
3) 可选：将分号 CSV 或逗号 CSV 进一步输出为 Parquet（推荐中间格式）

用法（命令行）：
  python 00xml转csv.py --xml2csv_py "...\xml2csv.py" --fcd_xml "...\fcd.xml" --out_dir "...\out" --prefix "FCD050"

也支持不传参数直接运行：会在终端提示输入。

输出默认：
- {out_dir}/{prefix}_fcd_semicolon.csv    （xml2csv 工具生成）
- {out_dir}/{prefix}_fcd.csv              （标准逗号 CSV，可选）
- {out_dir}/{prefix}_fcd.parquet          （Parquet，可选，推荐）

说明：
- xml2csv 工具可能依赖它所在目录的其他文件，所以建议 cwd 指向 xml2csv.py 所在目录。
"""

from __future__ import annotations

import os
import sys
import subprocess
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


# =========================
# 配置
# =========================
@dataclass
class Xml2CsvConfig:
    xml2csv_py: str
    fcd_xml: str
    out_dir: str
    prefix: str = "FCD"

    # 转换控制
    make_comma_csv: bool = True
    make_parquet: bool = False  # 推荐 True：后续步骤读 parquet 更快更省空间

    # 分块大小（越大越快，但更吃内存）
    chunksize: int = 500_000

    # 强制指定运行 xml2csv 的 cwd（默认自动取 xml2csv_py 所在目录）
    tool_cwd: Optional[str] = None

    # 输出编码
    csv_encoding: str = "utf-8-sig"


# =========================
# 工具函数
# =========================
def _ensure_file(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _run_xml2csv_tool(cfg: Xml2CsvConfig) -> str:
    """
    调用 xml2csv.py 输出分号分隔 CSV。
    返回：生成的分号 CSV 路径（..._semicolon.csv）
    """
    _ensure_file(cfg.xml2csv_py, "xml2csv_py")
    _ensure_file(cfg.fcd_xml, "fcd_xml")
    _safe_mkdir(cfg.out_dir)

    out_base = os.path.join(cfg.out_dir, f"{cfg.prefix}_fcd_semicolon")
    out_csv = out_base + ".csv"

    tool_cwd = cfg.tool_cwd or os.path.dirname(cfg.xml2csv_py) or None

    cmd = [
        sys.executable,
        cfg.xml2csv_py,
        "-o", out_base,
        "-s", ";",
        "-q", "",
        cfg.fcd_xml
    ]

    print("\n" + "=" * 80)
    print("[STEP00] xml2csv: FCD.xml -> semicolon CSV")
    print("CMD:", " ".join(cmd))
    print("CWD:", tool_cwd or "(default)")
    print("=" * 80)

    t0 = time.time()
    subprocess.run(cmd, check=True, text=True, encoding="utf-8", cwd=tool_cwd)
    dt = time.time() - t0

    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"xml2csv did not generate expected file: {out_csv}")

    print(f"[OK] semicolon CSV: {out_csv}   (time={dt:.1f}s)")
    return out_csv


def semicolon_to_comma_csv(in_semicolon_csv: str, out_comma_csv: str, chunksize: int, encoding: str) -> str:
    """
    将 ; 分隔的 CSV 转换为标准逗号 CSV。
    采用 chunksize 分块，避免爆内存。
    """
    _ensure_file(in_semicolon_csv, "in_semicolon_csv")
    _safe_mkdir(os.path.dirname(out_comma_csv))

    print("\n" + "=" * 80)
    print("[STEP00] semicolon CSV -> comma CSV (chunked)")
    print("IN :", in_semicolon_csv)
    print("OUT:", out_comma_csv)
    print(f"chunksize={chunksize}")
    print("=" * 80)

    first = True
    total_rows = 0
    t0 = time.time()

    # sep=";" 建议 engine="c"（更快），但遇到异常格式可改 python
    reader = pd.read_csv(in_semicolon_csv, sep=";", engine="c", chunksize=chunksize)

    for chunk in reader:
        total_rows += len(chunk)
        chunk.to_csv(
            out_comma_csv,
            index=False,
            encoding=encoding,
            mode=("w" if first else "a"),
            header=first
        )
        first = False

    dt = time.time() - t0
    print(f"[OK] comma CSV: {out_comma_csv}   rows={total_rows:,}   (time={dt:.1f}s)")
    return out_comma_csv


def to_parquet(in_csv: str, out_parquet: str, sep: Optional[str], chunksize: int) -> str:
    """
    CSV -> Parquet（可分块）
    - 若 CSV 很大，推荐用分块 + concat（会有一定内存峰值）
    - 如果你装了 pyarrow，to_parquet 会更快
    """
    _ensure_file(in_csv, "in_csv")
    _safe_mkdir(os.path.dirname(out_parquet))

    print("\n" + "=" * 80)
    print("[STEP00] CSV -> Parquet")
    print("IN :", in_csv)
    print("OUT:", out_parquet)
    print(f"chunksize={chunksize}")
    print("=" * 80)

    t0 = time.time()
    parts = []
    total_rows = 0

    # 如果文件不是逗号 CSV，可传 sep=";"；否则 sep=None 用默认逗号
    read_kwargs = {}
    if sep is not None:
        read_kwargs["sep"] = sep
        read_kwargs["engine"] = "c"

    for chunk in pd.read_csv(in_csv, chunksize=chunksize, **read_kwargs):
        total_rows += len(chunk)
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_parquet(out_parquet, index=False)

    dt = time.time() - t0
    print(f"[OK] parquet: {out_parquet}   rows={total_rows:,}   (time={dt:.1f}s)")
    return out_parquet


# =========================
# 对外：run()
# =========================
def run(cfg: Xml2CsvConfig) -> Tuple[str, Optional[str], Optional[str]]:
    """
    执行 step00
    返回：(semicolon_csv, comma_csv_or_None, parquet_or_None)
    """
    semicolon_csv = _run_xml2csv_tool(cfg)

    comma_csv = None
    if cfg.make_comma_csv:
        comma_csv = os.path.join(cfg.out_dir, f"{cfg.prefix}_fcd.csv")
        comma_csv = semicolon_to_comma_csv(
            in_semicolon_csv=semicolon_csv,
            out_comma_csv=comma_csv,
            chunksize=cfg.chunksize,
            encoding=cfg.csv_encoding
        )

    parquet_path = None
    if cfg.make_parquet:
        # 优先把逗号 CSV 转 parquet（字段更标准）；若没生成逗号 CSV，则用分号 CSV
        src = comma_csv or semicolon_csv
        sep = None if comma_csv else ";"  # 只有源是分号 CSV 才需要 sep=";"
        parquet_path = os.path.join(cfg.out_dir, f"{cfg.prefix}_fcd.parquet")
        parquet_path = to_parquet(src, parquet_path, sep=sep, chunksize=cfg.chunksize)

    return semicolon_csv, comma_csv, parquet_path


# =========================
# CLI / 交互
# =========================
def _ask(prompt: str, default: Optional[str] = None) -> str:
    tip = f" (default: {default})" if default else ""
    s = input(f"{prompt}{tip}\n> ").strip().strip('"').strip("'")
    return s if s else (default or "")


def main():
    p = argparse.ArgumentParser(description="FCD.xml -> CSV/Parquet (via xml2csv tool)")
    p.add_argument("--xml2csv_py", default="", help="path to xml2csv.py")
    p.add_argument("--fcd_xml", default="", help="path to fcd.xml")
    p.add_argument("--out_dir", default="", help="output directory")
    p.add_argument("--prefix", default="FCD", help="output filename prefix")

    p.add_argument("--no_comma", action="store_true", help="do not convert to comma csv")
    p.add_argument("--parquet", action="store_true", help="also export parquet")
    p.add_argument("--chunksize", type=int, default=500_000, help="chunk size for pandas read")
    p.add_argument("--tool_cwd", default="", help="cwd for running xml2csv (optional)")

    args = p.parse_args()

    # 如果用户没传参数，就走最简交互
    if not args.xml2csv_py:
        args.xml2csv_py = _ask("Enter path to xml2csv.py", default=args.xml2csv_py)
    if not args.fcd_xml:
        args.fcd_xml = _ask("Enter path to fcd.xml", default=args.fcd_xml)
    if not args.out_dir:
        args.out_dir = _ask("Enter output directory (out_dir)", default=os.getcwd())

    cfg = Xml2CsvConfig(
        xml2csv_py=args.xml2csv_py,
        fcd_xml=args.fcd_xml,
        out_dir=args.out_dir,
        prefix=args.prefix,
        make_comma_csv=(not args.no_comma),
        make_parquet=args.parquet,
        chunksize=args.chunksize,
        tool_cwd=(args.tool_cwd if args.tool_cwd else None),
    )

    semicolon_csv, comma_csv, parquet_path = run(cfg)

    print("\n✅ Step00 finished")
    print("semicolon CSV:", semicolon_csv)
    print("comma CSV    :", comma_csv or "(skipped)")
    print("parquet      :", parquet_path or "(skipped)")


if __name__ == "__main__":
    main()

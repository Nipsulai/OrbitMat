#!/usr/bin/env python3
"""
SLURM array job wrapper for the CP2K pipeline.

Each array task processes a contiguous slice [--start, --end) of the input JSON,
writing results to its own output directory.  No interactive prompts.

Usage (called from submit.sbatch):
    python calc/hpc/run_slurm.py calc/config.json \
        --start $START_IDX --end $END_IDX \
        --out /scratch/results/tzvp_${SLURM_ARRAY_TASK_ID}
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Make calc/src importable regardless of working directory
_CALC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CALC))
sys.path.insert(0, str(_CALC.parent))

from src.config import METHODS
from src.execute import CP2KExecutor
from src.workflow import CP2KWorkflow


def _load_chunk(input_json: str, start: int, end: int | None) -> list:
    with open(input_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    chunk = data[start:end]
    if not chunk:
        print(f"[hpc] No entries in slice [{start}:{end}] (total={len(data)}). Exiting.")
        sys.exit(0)
    print(f"[hpc] Slice [{start}:{end}]: {len(chunk)} entries (total={len(data)})")
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SLURM array job wrapper for run_cp2k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="JSON config file (same format as run_cp2k.py)")
    parser.add_argument("--start", type=int, default=0,
                        help="First entry index (inclusive)")
    parser.add_argument("--end",   type=int, default=None,
                        help="Last entry index (exclusive); default = end of list")
    parser.add_argument("--out",   default=None,
                        help="Output directory override (recommended: include $SLURM_ARRAY_TASK_ID)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg_dict = json.load(f)
    # Strip comment keys
    cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}

    method     = cfg_dict["method"]
    parallel   = cfg_dict.get("parallel", 8)
    threads    = cfg_dict.get("threads", 4)
    dim        = cfg_dict.get("dim", "3D")
    rspace     = not cfg_dict.get("kspace", False)
    band       = cfg_dict.get("band", False)
    include_unk = cfg_dict.get("include_unk", False)
    use_uks    = cfg_dict.get("use_uks", False)
    log_level  = cfg_dict.get("log_level", "INFO")

    mcfg = METHODS[method]

    # Template: explicit path or derive from method
    template = cfg_dict.get("template") or f"calc/src/input/template/{mcfg.template}.inp"
    if not Path(template).exists():
        print(f"[hpc] Template not found: {template}")
        sys.exit(1)

    # Output: CLI flag > config field > auto-generated
    if args.out:
        output = args.out
    elif cfg_dict.get("output"):
        output = cfg_dict["output"]
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%m%d_%H%M")
        task_id = __import__("os").getenv("SLURM_ARRAY_TASK_ID", "0")
        output = f"calc/out/{method}_{ts}_t{task_id}"

    # ── Slice input ───────────────────────────────────────────────────────────
    chunk = _load_chunk(cfg_dict["input"], args.start, args.end)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(chunk, tmp)
    tmp.close()

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        executor = CP2KExecutor(
            n_parallel=parallel,
            threads_per_job=threads,
            sym=dim,
            rspace=rspace,
        )
        workflow = CP2KWorkflow(
            input_list_json=tmp.name,
            base_output_dir=output,
            template_file=template,
            executor=executor,
            method=method,
            sym=dim,
            rspace=rspace,
            band=band,
            skip_if_unk=not include_unk if mcfg.has_dft else False,
            use_uks=use_uks,
            log_level=log_level,
        )
        workflow.run()
        print(f"[hpc] Done. Results: {output}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CP2K high-throughput workflow runner.

Usage:
    python run_cp2k.py --input structures.json --method pbe
    python run_cp2k.py --input structures.json --method xtb
    python run_cp2k.py --input molecules.json --method xyz
    python run_cp2k.py --input structures.json --method pbe --parallel 4 --threads 8
    python run_cp2k.py --input structures.json --method pbe --include-unk
    python run_cp2k.py --restart calc/out/pbe/ --input data.json --method pbe
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import h5py
import pandas as pd
import numpy as np

from src.workflow import CP2KWorkflow
from src.execute import CP2KExecutor
from src.config import VALID_METHODS


def collect_restart_jobs(restart_dir: Path, input_json: str, method: str) -> list:
    """Scan restart_dir for subfolders that need to be run.

    Skips subfolders that contain:
      - matrices.pkl  (successful run)
      - out.cp2k      (attempted but did not converge)
    #TODO: Restart the unconverged structures from .kp files
    Runs subfolders that have input.inp but neither of the above.
    """
    restart_dir = Path(restart_dir)
    if not restart_dir.is_dir():
        print(f"Error: Restart directory not found: {restart_dir}")
        sys.exit(1)

    # Load JSON to get bandgap_info
    with open(input_json, "r") as f:
        input_data = json.load(f)
    if isinstance(input_data, dict):
        input_data = list(input_data.values())

    entry_by_id = {}
    for entry in input_data:
        struct_id = Path(entry["cif_path"]).stem
        entry_by_id[struct_id] = entry

    jobs = []
    skipped_done = 0
    skipped_failed = 0

    for sub in sorted(restart_dir.iterdir()):
        if not sub.is_dir():
            continue

        input_file = sub / "input.inp"
        if not input_file.exists():
            continue

        if (sub / "matrices.pkl").exists():
            skipped_done += 1
            continue

        if (sub / "out.cp2k").exists():
            skipped_failed += 1
            continue

        entry = entry_by_id.get(sub.name)
        jobs.append({
            "job_id": sub.name,
            "input_path": entry["cif_path"] if entry else None,
            "input_file": str(input_file.resolve()),
            "work_dir": str(sub.resolve()),
            "method": method,
            "bandgap_info": entry.get("bandgap_info") if entry else None,
        })

    print(f"Restart scan: {skipped_done} done, {skipped_failed} failed/skipped, {len(jobs)} to run")
    return jobs


def main():
    parser = argparse.ArgumentParser(description="CP2K Calculator")

    parser.add_argument("--input", required=True, help="JSON file with input paths (CIF or XYZ)")
    parser.add_argument("--restart", default=None, help="Output folder of a previous run to restart incomplete jobs")
    parser.add_argument("--method", required=True, choices=VALID_METHODS)
    parser.add_argument("--sym", default="XYZ", choices=["XYZ", "XY"], help="Symmetry of crystals")
    parser.add_argument("--template", default=None, help="CP2K input template")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--parallel", type=int, default=8, help="Parallel jobs")
    parser.add_argument("--threads", type=int, default=4, help="Threads per job")
    parser.add_argument("--include-unk", action="store_true", help="Include UKS (PBE only)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")

    args = parser.parse_args()

    if args.method in {"xyz", "xtb"}:
        args.parallel = 32
        args.threads = 1

    total_threads = args.parallel * args.threads
    if total_threads > 32:
        print(f"Warning: Total threads ({total_threads}) exceeds 32")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"Error: Input list not found: {args.input}")
        sys.exit(1)

    # Whether continue previoous calculation
    if args.restart is not None:
        restart_dir = Path(args.restart)
        jobs = collect_restart_jobs(restart_dir, args.input, args.method)

        if not jobs:
            print("Nothing to run.")
            sys.exit(0)

        print(f"""
            ╔═════════════════════════════════════════╗
            ║       CP2K Restart Calculation          ║
            ╠═════════════════════════════════════════╣
            ║  Method:          {args.method.upper():<19}║
            ║  Restart dir:     {restart_dir.name:<19}║
            ║  Jobs to run:     {len(jobs):<19}║
            ║  Parallel:        {args.parallel:<19}║
            ║  Threads/job:     {args.threads:<19}║
            ║  Total threads:   {total_threads:<19}║
            ╚═════════════════════════════════════════╝
            """)

        response = input("Proceed? (y/n): ")
        if response.lower() != "y":
            print("Cancelled")
            sys.exit(0)

        executor = CP2KExecutor(
            n_parallel=args.parallel,
            threads_per_job=args.threads,
            sym=args.sym,
        )

        results = executor.execute_batch(jobs)

        n_success = sum(1 for r in results if r["success"])
        n_failed = len(results) - n_success
        print(f"\n Restart complete. {n_success} succeeded, {n_failed} failed.")
        sys.exit(0)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output = f"calc/out/{args.method}_{timestamp}"

    if args.template is None:
        args.template = f"calc/src/input/template_{args.method}.inp"
        if not Path(args.template).exists():
            print(f"Error: Template not found: {args.template}")
            sys.exit(1)

    input_type = "XYZ" if args.method == "xyz" else "CIF"
    uks_line = f"║  Include UKS:     {str(args.include_unk):<19}║\n" if args.method == "pbe" else ""

    print(f"""
            ╔═════════════════════════════════════════╗
            ║       CP2K Calculator                   ║
            ╠═════════════════════════════════════════╣
            ║  Method:          {args.method.upper():<19}║
            ║  Input type:      {input_type:<19}║
            ║  Input list:      {Path(args.input).name:<19}║
            ║  Template:        {args.template:<19}║
            ║  Output:          {Path(args.output).name:<19}║
            ║  Parallel:        {args.parallel:<19}║
            ║  Threads/job:     {args.threads:<19}║
            ║  Total threads:   {total_threads:<19}║
            {uks_line}╚═════════════════════════════════════════╝
            """)

    response = input("Proceed? (y/n): ")
    if response.lower() != "y":
        print("Cancelled")
        sys.exit(0)

    executor = CP2KExecutor(
        n_parallel=args.parallel,
        threads_per_job=args.threads,
        sym = args.sym
    )

    workflow = CP2KWorkflow(
        input_list_json=args.input,
        base_output_dir=args.output,
        template_file=args.template,
        executor=executor,
        method=args.method,
        sym = args.sym,
        skip_if_unk=not args.include_unk if args.method == "pbe" else False,
        log_level=args.log_level,
    )

    workflow.run()

    print(f"\n Complete. Results: {args.output}")


if __name__ == "__main__":
    main()
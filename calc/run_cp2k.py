#!/usr/bin/env python3
"""
Runs CP2K calculations.

Usage:
    python run_cp2k.py config.json
    python run_cp2k.py config.json --restart calc/out/dz_0401_1200/
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.workflow import CP2KWorkflow
from src.execute import CP2KExecutor
from src.config import METHODS


# ── Run configuration ─────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    method: str
    input: str
    output: Optional[str]   = None
    parallel: int           = 8
    threads: int            = 4
    dim: str                = "3D"
    template: Optional[str] = None
    include_unk: bool       = False
    kspace: bool            = False
    band: bool              = False
    use_uks: bool           = False
    log_level: str          = "INFO"

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        with open(path) as f:
            data = json.load(f)
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @property
    def rspace(self) -> bool:
        return not self.kspace

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method: '{self.method}'. Valid: {list(METHODS)}")
        mcfg = METHODS[self.method]
        if self.band and not mcfg.supports_band:
            raise ValueError(f"band=true not supported for method '{self.method}'")
        if self.dim not in ("3D", "2D"):
            raise ValueError(f"dim must be '3D' or '2D', got '{self.dim}'")
        if self.log_level not in ("DEBUG", "INFO", "WARNING"):
            raise ValueError(f"log_level must be DEBUG/INFO/WARNING, got '{self.log_level}'")
        if self.parallel * self.threads > 32:
            raise ValueError(
                f"parallel ({self.parallel}) * threads ({self.threads}) = "
                f"{self.parallel * self.threads} exceeds 32"
            )
        if not Path(self.input).exists():
            raise FileNotFoundError(f"Input file not found: {self.input}")

    def resolve_template(self) -> str:
        if self.template is not None:
            if not Path(self.template).exists():
                raise FileNotFoundError(f"Template not found: {self.template}")
            return self.template
        stem = METHODS[self.method].template
        path = f"calc/src/input/template/{stem}.inp"
        if not Path(path).exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path

    def resolve_output(self) -> str:
        if self.output is not None:
            return self.output
        timestamp = datetime.now().strftime("%m%d_%H%M")
        return f"calc/out/{self.method}_{timestamp}"


# ── Restart helper ────────────────────────────────────────────────────────────

def collect_restart_jobs(restart_dir: Path, input_json: str, method: str) -> list:
    """Scan restart_dir for subfolders that still need to be run.

    Skips:   matrices.pkl  (success)  |  out.cp2k  (attempted, did not converge)
    Runs:    has input.inp but neither of the above
    """
    restart_dir = Path(restart_dir)
    if not restart_dir.is_dir():
        print(f"Error: Restart directory not found: {restart_dir}")
        sys.exit(1)

    with open(input_json) as f:
        input_data = json.load(f)
    if isinstance(input_data, dict):
        input_data = list(input_data.values())

    entry_by_id = {Path(e["cif_path"]).stem: e for e in input_data}

    jobs = []
    skipped_done = skipped_failed = 0

    for sub in sorted(restart_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "input.inp").exists():
            continue
        if (sub / "matrices.pkl").exists():
            skipped_done += 1
            continue
        if (sub / "out.cp2k").exists():
            skipped_failed += 1
            continue

        entry = entry_by_id.get(sub.name)
        jobs.append({
            "job_id":      sub.name,
            "input_path":  entry["cif_path"] if entry else None,
            "input_file":  str((sub / "input.inp").resolve()),
            "work_dir":    str(sub.resolve()),
            "method":      method,
            "bandgap_info": entry.get("bandgap_info") if entry else None,
        })

    print(f"Restart scan: {skipped_done} done, {skipped_failed} failed/skipped, {len(jobs)} to run")
    return jobs


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Banner ────────────────────────────────────────────────────────────────────
_RAINBOW = ["\033[91m", "\033[93m", "\033[92m", "\033[96m", "\033[94m", "\033[95m"]
_RESET   = "\033[0m"
_W, _VW  = 41, 19   # interior width, value column width


def _banner(title: str, rows: list) -> None:
    lw = _W - _VW  # label column width = 22
    lines = [
        f"╔{'═' * _W}╗",
        f"║{title:^{_W}}║",
        f"╠{'═' * _W}╣",
        *(f"║  {lbl:<{lw - 2}}{str(val):<{_VW}}║" for lbl, val in rows),
        f"╚{'═' * _W}╝",
    ]
    print()
    for i, line in enumerate(lines):
        print(f"            {_RAINBOW[i % len(_RAINBOW)]}{line}{_RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CP2K Calculator")
    parser.add_argument("config", help="JSON config file")
    parser.add_argument("--restart", default=None,
                        help="Output folder of a previous run to restart incomplete jobs")
    args = parser.parse_args()

    try:
        cfg = RunConfig.from_json(Path(args.config))
    except (ValueError, FileNotFoundError) as e:
        print(f"Config error: {e}")
        sys.exit(1)

    template  = cfg.resolve_template()
    output    = cfg.resolve_output()
    total_threads = cfg.parallel * cfg.threads
    mcfg      = METHODS[cfg.method]

    # ── Restart path ─────────────────────────────────────────────────────────
    if args.restart is not None:
        jobs = collect_restart_jobs(Path(args.restart), cfg.input, cfg.method)
        if not jobs:
            print("Nothing to run.")
            sys.exit(0)

        _banner("CP2K Restart Calculation", [
            ("Method:",       cfg.method.upper()),
            ("Restart dir:",  Path(args.restart).name),
            ("Jobs to run:",  len(jobs)),
            ("Parallel:",     cfg.parallel),
            ("Threads/job:",  cfg.threads),
            ("Total threads:", total_threads),
        ])

        if input("Proceed? (y/n): ").lower() != "y":
            print("Cancelled")
            sys.exit(0)

        executor = CP2KExecutor(
            n_parallel=cfg.parallel,
            threads_per_job=cfg.threads,
            sym=cfg.dim,
            rspace=cfg.rspace,
        )
        results = executor.execute_batch(jobs)
        n_ok = sum(1 for r in results if r["success"])
        print(f"\nRestart complete. {n_ok} succeeded, {len(results)-n_ok} failed.")
        sys.exit(0)

    # ── Normal run ────────────────────────────────────────────────────────────
    input_type = "XYZ" if not mcfg.periodic else "CIF"
    space_type = "R-space" if cfg.rspace else "K-space"
    rows = [
        ("Method:",        cfg.method.upper()),
        ("Matrices:",      space_type),
        ("Input type:",    input_type),
        ("Input list:",    Path(cfg.input).name),
        ("Template:",      Path(template).name),
        ("Output:",        Path(output).name),
        ("Parallel:",      cfg.parallel),
        ("Threads/job:",   cfg.threads),
        ("Total threads:", total_threads),
    ]
    if mcfg.has_dft:
        rows.append(("Include UKS:", cfg.include_unk))
    if cfg.band:
        rows.append(("Band structure:", cfg.band))
    _banner("CP2K Calculator", rows)

    if input("Proceed? (y/n): ").lower() != "y":
        print("Cancelled")
        sys.exit(0)

    executor = CP2KExecutor(
        n_parallel=cfg.parallel,
        threads_per_job=cfg.threads,
        sym=cfg.dim,
        rspace=cfg.rspace,
    )

    workflow = CP2KWorkflow(
        input_list_json=cfg.input,
        base_output_dir=output,
        template_file=template,
        executor=executor,
        method=cfg.method,
        sym=cfg.dim,
        rspace=cfg.rspace,
        band=cfg.band,
        skip_if_unk=not cfg.include_unk if mcfg.has_dft else False,
        use_uks=cfg.use_uks,
        log_level=cfg.log_level,
    )

    workflow.run()
    print(f"\nComplete. Results: {output}")


if __name__ == "__main__":
    main()

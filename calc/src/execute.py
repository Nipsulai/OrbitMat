"""CP2K executor with parallel processing and live progress tracking."""

import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List
import glob

from tqdm import tqdm

from .config import OUTPUT, SSMP, SOURCE, TIMEOUT
from .postproc import postproc_matrices

def _cleanup_restart_files(work_dir: Path) -> None:
    """Delete CP2K restart files (.kp* and .wfn*) produced by xTB calculations."""
    for pattern in ["*.kp*", "*.wfn*"]:
        for f in glob.glob(str(work_dir / pattern)):
            Path(f).unlink(missing_ok=True)


def _run_cp2k_subprocess(cmd: str, work_dir: Path, env: dict, timeout: int) -> float:
    """Run a CP2K subprocess. Returns elapsed seconds. Raises TimeoutExpired on timeout."""
    start = time.time()
    subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        cwd=work_dir,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return time.time() - start


def run_single_calculation(
    input_file: str,
    work_dir: str,
    job_id: str,
    threads: int,
    method: str,
    sym,
    rspace: bool = True,
    bandgap_info: Dict = None,
    energy_dft_ha: float = None,
) -> Dict:
    work_dir = Path(work_dir)
    result = {
        "_work_dir": str(Path(work_dir).resolve()),
        "success": False,
        "scf_converged": None,
        "runtime_seconds": None,
        "error": None,
    }

    Path(work_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = "1"

    if method == "charge_xtb":
        return _run_charge_xtb(
            result, input_file, work_dir, env, sym, rspace, bandgap_info, energy_dft_ha
        )

    try:
        input_file_abs = str(Path(input_file).absolute())
        cmd = f"source {SOURCE} && {SSMP} -i {input_file_abs} -o {OUTPUT}"

        result["runtime_seconds"] = _run_cp2k_subprocess(cmd, work_dir, env, TIMEOUT)

        # Clean up restart files
        _cleanup_restart_files(work_dir)

        output_log = Path(work_dir) / OUTPUT
        if output_log.exists():
            output_text = output_log.read_text()

            if "SCF run converged" in output_text:
                result["scf_converged"] = True
                result["success"] = True
                _ = postproc_matrices(work_dir, method, sym, rspace=rspace, bandgap_info=bandgap_info, energy_dft_ha=energy_dft_ha)

            # When using xtb some structures may lead to unphysical charges
            elif "Switch-off CHECK_ATOMIC_CHARGES keyword" in output_text:
                result["scf_converged"] = False
                result["error"] = "Unphysical ATOMIC CHARGES"
                print(f"Unphysical ATOMIC CHARGES: {work_dir}")

            elif "SCF run NOT converged" in output_text:
                result["scf_converged"] = False
                result["error"] = "Max SCF iterations"
                print(f"Max SCF iterations: {work_dir}")
            else:
                result["scf_converged"] = False
                result["error"] = "Unknown convergence status"
                print(f"Error: {work_dir}")

            if any(kw in output_text for kw in ["ERROR", "ABORT", "FATAL", "SIGSEGV"]):
                result["success"] = False
                result["error"] = result.get("error") or "Runtime error"
                print(f"Error: {work_dir}")
        else:
            result["error"] = "No output file generated"
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout (>{TIMEOUT}s)"
        print(f"Timeout (>{TIMEOUT}s): {work_dir}")
    except Exception as e:
        result["error"] = str(e)

    return result


def _run_charge_xtb(
    result: Dict,
    input_file: str,
    work_dir: Path,
    env: dict,
    sym,
    rspace: bool,
    bandgap_info: Dict,
    energy_dft_ha: float,
) -> Dict:
    """
    Two-step xTB calculation that handles unphysical Mulliken charges:
      Step 1 — CHECK_ATOMIC_CHARGES F: converge the SCF freely, produces restart WFN.
      Step 2 — CHECK_ATOMIC_CHARGES T + SCF_GUESS RESTART: re-run from warm start
               with the charge check enabled; writes matrices.
    Restart files (*.kp*, *.wfn*) are deleted after step 2.
    """
    inp_text = Path(input_file).read_text()

    # ── Step 1: no charge check ──────────────────────────────────────────────
    nocheck_inp = work_dir / "input_nocheck.inp"
    nocheck_inp.write_text(
        inp_text.replace("CHECK_ATOMIC_CHARGES T", "CHECK_ATOMIC_CHARGES F")
    )
    cmd1 = f"source {SOURCE} && {SSMP} -i {nocheck_inp.name} -o out_nocheck.cp2k"

    try:
        runtime1 = _run_cp2k_subprocess(cmd1, work_dir, env, TIMEOUT)
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout step1 (>{TIMEOUT}s)"
        print(f"Timeout step1 (>{TIMEOUT}s): {work_dir}")
        _cleanup_restart_files(work_dir)
        return result
    except Exception as e:
        result["error"] = str(e)
        _cleanup_restart_files(work_dir)
        return result

    nocheck_out = work_dir / "out_nocheck.cp2k"
    if not nocheck_out.exists():
        result["error"] = "No output from step 1"
        _cleanup_restart_files(work_dir)
        return result

    nocheck_text = nocheck_out.read_text()
    if "SCF run converged" not in nocheck_text:
        if "SCF run NOT converged" in nocheck_text:
            result["error"] = "Step1: Max SCF iterations"
        else:
            result["error"] = "Step1: Unknown convergence status"
        result["scf_converged"] = False
        print(f"charge_xtb step1 failed ({result['error']}): {work_dir}")
        _cleanup_restart_files(work_dir)
        return result

    # ── Step 2: charge check enabled, warm-start from step-1 WFN ────────────
    check_inp = work_dir / "input_check.inp"
    check_inp.write_text(
        inp_text.replace("SCF_GUESS ATOMIC", "SCF_GUESS RESTART")
    )
    cmd2 = f"source {SOURCE} && {SSMP} -i {check_inp.name} -o {OUTPUT}"

    try:
        runtime2 = _run_cp2k_subprocess(cmd2, work_dir, env, TIMEOUT)
        result["runtime_seconds"] = runtime1 + runtime2
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout step2 (>{TIMEOUT}s)"
        result["runtime_seconds"] = runtime1 + TIMEOUT
        print(f"Timeout step2 (>{TIMEOUT}s): {work_dir}")
        return result
    except Exception as e:
        result["error"] = str(e)
        return result
    finally:
        # Always clean up restart files after step 2 regardless of outcome
        _cleanup_restart_files(work_dir)

    output_log = work_dir / OUTPUT
    if not output_log.exists():
        result["error"] = "No output from step 2"
        return result

    output_text = output_log.read_text()

    if any(kw in output_text for kw in ["ERROR", "ABORT", "FATAL", "SIGSEGV"]):
        result["success"] = False
        result["error"] = "Runtime error in step 2"
        print(f"Error step2: {work_dir}")
        return result

    if "SCF run converged" in output_text:
        result["scf_converged"] = True
        result["success"] = True
        _ = postproc_matrices(
            work_dir, "xtb", sym, rspace=rspace,
            bandgap_info=bandgap_info, energy_dft_ha=energy_dft_ha,
        )
    elif "Switch-off CHECK_ATOMIC_CHARGES keyword" in output_text:
        result["scf_converged"] = False
        result["error"] = "Step2: Unphysical ATOMIC CHARGES (even after warm start)"
        print(f"charge_xtb step2 charge error: {work_dir}")
    elif "SCF run NOT converged" in output_text:
        result["scf_converged"] = False
        result["error"] = "Step2: Max SCF iterations"
        print(f"charge_xtb step2 not converged: {work_dir}")
    else:
        result["scf_converged"] = False
        result["error"] = "Step2: Unknown convergence status"
        print(f"charge_xtb step2 unknown status: {work_dir}")

    return result


class CP2KExecutor:
    def __init__(self, n_parallel: int, threads_per_job: int, sym, rspace: bool = True):
        self.n_parallel = n_parallel
        self.threads_per_job = threads_per_job
        self.sym = sym
        self.rspace = rspace

    def execute_batch(self, jobs: List[Dict]) -> List[Dict]:
        results = []
        n_success = 0
        n_failed = 0

        with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
            futures = {
                executor.submit(
                    run_single_calculation,
                    job["input_file"],
                    job["work_dir"],
                    job["job_id"],
                    self.threads_per_job,
                    job["method"],
                    self.sym,
                    job.get("rspace", self.rspace),
                    job.get("bandgap_info"),
                    job.get("energy_dft_ha"),
                ): job
                for job in jobs
            }

            with tqdm(
                total=len(jobs),
                desc="CP2K",
                unit="job",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✓{postfix}",
            ) as pbar:
                pbar.set_postfix_str(f"0 ✗0")

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)

                        if result["success"]:
                            n_success += 1
                        else:
                            n_failed += 1

                        pbar.set_postfix_str(f"{n_success} ✗{n_failed}")
                        pbar.update(1)

                    except Exception as e:
                        n_failed += 1
                        pbar.set_postfix_str(f"{n_success} ✗{n_failed}")
                        pbar.update(1)

        return results
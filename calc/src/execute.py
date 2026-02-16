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

def run_single_calculation(
    input_file: str,
    work_dir: str,
    job_id: str,
    threads: int,
    method: str,
    sym,
    bandgap_info: Dict = None
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

    try:
        start = time.time()

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = "1"

        input_file_abs = str(Path(input_file).absolute())
        cmd = f"source {SOURCE} && {SSMP} -i {input_file_abs} -o {OUTPUT}"

        subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            cwd=work_dir,
            capture_output=True,
            timeout=TIMEOUT,
            env=env,
        )

        result["runtime_seconds"] = time.time() - start

        # Clean up .kp restart files
        for kp_file in glob.glob(str(work_dir / "*.kp*")):
            Path(kp_file).unlink(missing_ok=True)

        output_log = Path(work_dir) / OUTPUT
        if output_log.exists():
            output_text = output_log.read_text()

            if "SCF run converged" in output_text:
                result["scf_converged"] = True
                result["success"] = True
                _ = postproc_matrices(work_dir, method, sym, bandgap_info=bandgap_info)

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
        # Keep .kp restart files for resuming
        #kp_files = glob.glob(str(work_dir / "*.kp*"))
        #if kp_files:
        #    result["restart_file"] = kp_files[0]
    except Exception as e:
        result["error"] = str(e)

    return result


class CP2KExecutor:
    def __init__(self, n_parallel: int, threads_per_job: int, sym):
        self.n_parallel = n_parallel
        self.threads_per_job = threads_per_job
        self.sym = sym

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
                    job.get("bandgap_info")
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
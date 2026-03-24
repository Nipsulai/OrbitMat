"""CP2K executor with parallel processing and live progress tracking."""

import os
import signal
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List
import glob


def _folder_size_human(path: Path) -> str:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.0f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"

from tqdm import tqdm

from .config import OUTPUT, SSMP, SOURCE, TIMEOUT, METHODS
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

def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_job(job: Dict) -> Dict:
    return run_single_calculation(
        job["input_file"],
        job["work_dir"],
        job["job_id"],
        job["threads"],
        job["method"],
        job["sym"],
        job.get("rspace", True),
        job.get("bandgap_info"),
        job.get("energy_dft_ha"),
        job.get("band", False),
        job.get("input_path"),
    )


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
    band: bool = False,
    input_path: str = None,
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
                if band and METHODS[method].supports_band:
                    try:
                        from .bands import postproc_bands
                        matrices_pkl = str(Path(work_dir) / "matrices.pkl")
                        stats = postproc_bands(work_dir, input_path, matrices_pkl)
                        result["vbm_ev"] = stats["vbm_ev"]
                        result["bandgap_path_eV"] = stats["bandgap_path_eV"]
                    except Exception as e:
                        result["band_error"] = str(e)
                        print(f"Band postproc failed: {work_dir}: {e}")

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

class CP2KExecutor:
    def __init__(self, n_parallel: int, threads_per_job: int, sym, rspace: bool = True):
        self.n_parallel = n_parallel
        self.threads_per_job = threads_per_job
        self.sym = sym
        self.rspace = rspace

    def execute_batch(self, jobs: List[Dict], result_callback=None) -> List[Dict]:
        # Embed per-job constants so _run_job receives a single picklable dict
        for job in jobs:
            job["threads"] = self.threads_per_job
            job["sym"] = self.sym
            job.setdefault("rspace", self.rspace)

        results = []
        n_success = 0
        n_failed = 0

        output_dir = Path(jobs[0]["work_dir"]).parent if jobs else None

        pool = Pool(processes=self.n_parallel, initializer=_worker_init)
        try:
            imap = pool.imap_unordered(_run_job, jobs, chunksize=1)
            with tqdm(
                total=len(jobs),
                desc="CP2K",
                unit="job",
                position=0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✓{postfix}",
            ) as pbar:
                with tqdm(
                    total=0,
                    bar_format="{desc}",
                    desc=f"  📁 {output_dir.name}  0 B" if output_dir else "",
                    position=1,
                    leave=True,
                ) as dbar:
                    pbar.set_postfix_str("0 ✗0")
                    for result in imap:
                        results.append(result)
                        if result_callback is not None:
                            result_callback(result)
                        if result["success"]:
                            n_success += 1
                        else:
                            n_failed += 1
                        pbar.set_postfix_str(f"{n_success} ✗{n_failed}")
                        pbar.update(1)
                        if output_dir:
                            dbar.set_description_str(
                                f"  📁 {output_dir.name}  {_folder_size_human(output_dir)}"
                            )
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

        return results

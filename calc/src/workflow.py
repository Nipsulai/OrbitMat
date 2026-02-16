"""CP2K workflow orchestrator for high-throughput calculations."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .input_generator import CP2KInputGenerator
from .execute import CP2KExecutor
from .config import VALID_METHODS

class CP2KWorkflow:
    def __init__(
        self,
        input_list_json: str,
        base_output_dir: str,
        template_file: str,
        executor: CP2KExecutor,
        method: str,
        sym: str, 
        skip_if_unk: bool = True,
        log_level: str = "INFO",
    ):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.method = method.lower()

        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method: {method}. Must be one of {VALID_METHODS}")

        self.logger = self._setup_logger(log_level)

        with open(input_list_json, "r") as f:
            input_data = json.load(f)

        # Handle both dict (keyed by ID) and list formats
        if isinstance(input_data, dict):
            input_data = list(input_data.values())

        self.input_entries = input_data
        self.input_paths = [entry["cif_path"] for entry in input_data]
        self.bandgap_info = {
            entry["cif_path"]: entry.get("bandgap_info")
            for entry in input_data
        }
        self.bravais = {
            entry["cif_path"]: entry.get("bravais")
            for entry in input_data
        }

        file_type = "XYZ" if self.method == "xyz" else "CIF"
        self.logger.info(f"Loaded {len(self.input_paths)} {file_type} files")

        self.input_generator = CP2KInputGenerator(
            method=method,
            sym=sym,
            template_file=template_file,
            logger=self.logger,
        )

        self.executor = executor
        self.skip_if_unk = skip_if_unk if method == "pbe" else False
        self.metadata_db = {}

    def _setup_logger(self, level: str) -> logging.Logger:
        logger = logging.getLogger("cp2k_workflow")
        logger.setLevel(getattr(logging, level))

        log_file = self.base_output_dir / "workflow.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, level))

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level))

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def _get_structure_id(self, input_path: str) -> str:
        return Path(input_path).stem

    def generate_inputs(self) -> List[Dict]:
        jobs = []
        skipped_count = 0

        for input_path in self.input_paths:
            struct_id = self._get_structure_id(input_path)
            struct_dir = self.base_output_dir / struct_id
            struct_dir.mkdir(parents=True, exist_ok=True)

            input_file = struct_dir / "input.inp"
            metadata = self.input_generator.generate_input(
                input_path=input_path,
                output_path=str(input_file),
                bravais= self.bravais.get(input_path),
                skip_if_unk=self.skip_if_unk,
            )

            self.metadata_db[input_path] = metadata

            if metadata.get("skipped"):
                skipped_count += 1
                continue

            jobs.append({
                "job_id": struct_id,
                "input_path": input_path,
                "input_file": str(input_file),
                "work_dir": str(struct_dir),
                "method": self.method,
                "bandgap_info": self.bandgap_info.get(input_path),
            })

        self.logger.info(f"Generated {len(jobs)} inputs, skipped {skipped_count}")
        if not jobs:
            raise ValueError("No input files created")
        return jobs

    def merge_metadata(self, results: List[Dict]) -> None:
        for result in results:
            work_dir = result.pop("_work_dir")
            struct_id = Path(work_dir).name
            for path in self.input_paths:
                if self._get_structure_id(path) == struct_id:
                    if path in self.metadata_db:
                        self.metadata_db[path]["_work_dir"] = work_dir
                        self.metadata_db[path].update(result)
                    else:
                        self.metadata_db[path] = {"_work_dir": work_dir, **result}
                    break

    def save_metadata(self) -> None:
        for input_path, metadata in self.metadata_db.items():
            struct_id = self._get_structure_id(input_path)
            work_dir = metadata.pop("_work_dir", self.base_output_dir / struct_id)

            with open(Path(work_dir) / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        successful = sum(1 for m in self.metadata_db.values() if m.get("success"))
        skipped = sum(1 for m in self.metadata_db.values() if m.get("skipped"))
        failed_paths = [
            str(Path(path).resolve())
            for path, m in self.metadata_db.items()
            if not m.get("success") and not m.get("skipped")
        ]

        summary = {
            "method": self.method,
            "total": len(self.input_paths),
            "successful": successful,
            "skipped": skipped,
            "failed": len(failed_paths),
            "failed_paths": failed_paths,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.base_output_dir / "0_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Done: {successful} ok, {len(failed_paths)} failed, {skipped} skipped")

    def run(self) -> None:
        self.logger.info(f"Starting workflow (method={self.method})")
        jobs = self.generate_inputs()
        results = self.executor.execute_batch(jobs)
        self.merge_metadata(results)
        self.save_metadata()
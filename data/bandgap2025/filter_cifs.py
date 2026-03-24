"""
Filter bandgap2025/all_data.json and produce a filtered data.json + log.yaml.

Adapted from data/scripts/filter_cifs.py for the bandgap2025 dataset.

Differences from the original:
  - bandgap_method: "hse" or "pbesol" (not "pbe")
  - No magnetic_ordering / soc / gap_type fields
  - natoms is a top-level field (not under "orig")
  - bandgap_info also contains energy, volume, density, cbm, vbm, e_fermi

Usage:
    python filter_cifs.py config.json
    python filter_cifs.py config.json -i all_data.json -o filtered/ --prefix run
"""
import json
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Set
from datetime import datetime


@dataclass
class FilterConfig:
    """
    Dataset filtering. Defaults to 'pass-all' unless specified.
    """

    exclude_lattice_types: Optional[Set[str]] = None   # e.g. {"triclinic"}
    exclude_centering_types: Optional[Set[str]] = None  # e.g. {"simple"}

    bandgap_min: float = 0.0
    bandgap_max: float = float('inf')
    bandgap_method: str = "hse"   # "hse" or "pbesol"

    n_atoms_min: Optional[int] = None
    n_atoms_max: Optional[int] = None

    insulators_only: bool = False   # True → only keep entries with bandgap > 0
    exclude_magnetic: bool = False  # True → drop entries with non-zero magmom

    # Keep only the top K results after sorting by atom count
    keep_k: Optional[int] = None

    def __post_init__(self):
        for field_name in ['exclude_lattice_types', 'exclude_centering_types']:
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, {val})
            elif isinstance(val, list):
                setattr(self, field_name, set(val))

    @classmethod
    def from_json(cls, path: Path) -> 'FilterConfig':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        if data.get('bandgap_max') == "inf":
            data['bandgap_max'] = float('inf')
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, set):
                d[key] = sorted(list(value))
            elif value == float('inf'):
                d[key] = "inf"
        return d


class DatasetFilter:
    def __init__(self, database_path: Path, config: FilterConfig):
        self.database_path = Path(database_path)
        self.config = config

        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")

        with open(self.database_path) as f:
            self.data = json.load(f)

    def _check_lattice(self, entry: dict) -> bool:
        if not self.config.exclude_lattice_types:
            return True
        return entry["bravais"]["lattice"] not in self.config.exclude_lattice_types

    def _check_centering(self, entry: dict) -> bool:
        if not self.config.exclude_centering_types:
            return True
        return entry["bravais"]["center"] not in self.config.exclude_centering_types

    def _check_bandgap(self, entry: dict) -> bool:
        bg_info = entry["bandgap_info"]
        val = bg_info.get(f"bandgap_{self.config.bandgap_method}")
        if val is None:
            return False
        if self.config.insulators_only and val <= 0.0:
            return False
        return self.config.bandgap_min <= val <= self.config.bandgap_max

    def _check_magmom(self, entry: dict) -> bool:
        if not self.config.exclude_magnetic:
            return True
        magmom = entry.get("magmom")
        return magmom is None or magmom == 0.0

    def _check_size(self, entry: dict) -> bool:
        n = entry["natoms"]
        if self.config.n_atoms_min is not None and n < self.config.n_atoms_min:
            return False
        if self.config.n_atoms_max is not None and n > self.config.n_atoms_max:
            return False
        return True

    def filter_dataset(self) -> List[dict]:
        checks = [
            self._check_lattice,
            self._check_centering,
            self._check_bandgap,
            self._check_size,
            self._check_magmom,
        ]
        return [entry for entry in self.data if all(f(entry) for f in checks)]

    def save_results(self, output_dir: Path, prefix: str = "run") -> Path:
        output_dir = Path(output_dir)
        filtered = self.filter_dataset()

        # Sort by atom count
        filtered.sort(key=lambda x: x["natoms"])

        if self.config.keep_k == -1:
            self.config.keep_k = len(filtered)

        if self.config.keep_k is not None and len(filtered) > self.config.keep_k:
            print(f"Limiting to top {self.config.keep_k} entries (sorted by size).")
            filtered = filtered[:self.config.keep_k]

        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_dir = output_dir / f"{prefix}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "data.json", 'w') as f:
            json.dump(filtered, f, indent=2)

        # Band gap stats for the filtered set
        bgs = [e["bandgap_info"][f"bandgap_{self.config.bandgap_method}"] for e in filtered
               if e["bandgap_info"].get(f"bandgap_{self.config.bandgap_method}") is not None]
        insulators = [b for b in bgs if b > 0]
        n_magnetic = sum(
            1 for e in self.data
            if e.get("magmom") is not None and e.get("magmom") != 0.0
        )

        log_data = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "database_source": str(self.database_path.absolute()),
            },
            "stats": {
                "total_entries": len(self.data),
                "magnetic_entries": n_magnetic,
                "filtered_entries": len(filtered),
                "retention_rate_percent": round(len(filtered) / len(self.data) * 100, 2) if self.data else 0,
                "insulators": len(insulators),
                "metals": len(filtered) - len(insulators),
                f"bandgap_{self.config.bandgap_method}_min": round(min(bgs), 4) if bgs else None,
                f"bandgap_{self.config.bandgap_method}_max": round(max(bgs), 4) if bgs else None,
                f"bandgap_{self.config.bandgap_method}_mean": round(sum(bgs) / len(bgs), 4) if bgs else None,
            },
            "configuration": self.config.to_dict(),
        }

        with open(save_dir / "log.yaml", 'w') as f:
            yaml.dump(log_data, f, sort_keys=False)

        print(f"\n--- Filtering Complete ---")
        print(f"Total entries:     {len(self.data)}")
        print(f"  Magnetic:        {n_magnetic}")
        print(f"Kept entries:      {len(filtered)}")
        print(f"  Insulators:      {len(insulators)}")
        print(f"  Metals:          {len(filtered) - len(insulators)}")
        print(f"Output directory:  {save_dir}")
        print(f"--------------------------\n")

        return save_dir


def main():
    parser = argparse.ArgumentParser(description="Filter bandgap2025 CIF dataset")
    parser.add_argument("config", type=Path, help="Path to JSON filter config")
    parser.add_argument("--database", "-i", type=Path,
                        default=Path(__file__).parent / "all_data.json",
                        help="Path to all_data.json")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path(__file__).parent / "filtered",
                        help="Output directory")
    parser.add_argument("--prefix", "-pre", type=str, default="run",
                        help="Prefix for the output folder name")

    args = parser.parse_args()
    config = FilterConfig.from_json(args.config)
    DatasetFilter(args.database, config).save_results(args.output, prefix=args.prefix)


if __name__ == "__main__":
    main()

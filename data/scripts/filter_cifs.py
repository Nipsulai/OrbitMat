import json
import yaml
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Set
from datetime import datetime

@dataclass
class FilterConfig:
    """
    Dataset filtering. Defaults to 'pass-all' filter unless specified.
    """

    magnetic_ordering: Optional[Set[str]] = None
    
    exclude_lattice_types: Optional[Set[str]] = None
    exclude_centering_types: Optional[Set[str]] = None
    
    bandgap_min: float = 0.0
    bandgap_max: float = float('inf')
    bandgap_method: str = "pbe"  # "pbe" or "hse"
    gap_type: Optional[Set[str]] = None
    
    n_atoms_min: Optional[int] = None
    n_atoms_max: Optional[int] = None
    
    soc: Optional[str] = None  # "True", "False", or None
    
    # Keep only the top K results after sorting
    keep_k: Optional[int] = None  

    def __post_init__(self):
        # Convert single strings to sets
        for field_name in ['magnetic_ordering', 'exclude_lattice_types', 
                           'exclude_centering_types', 'gap_type']:
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, {val})
            elif isinstance(val, list):
                setattr(self, field_name, set(val))

    @classmethod
    def from_json(cls, path: Path) -> 'FilterConfig':
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Handle 'inf' strings from JSON
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
    """
    Filter dataset based on the config.
    """
    def __init__(self, database_path: Path, config: FilterConfig):
        self.database_path = Path(database_path)
        self.config = config
        
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found at: {self.database_path}")

        with open(self.database_path, 'r') as f:
            self.data = json.load(f)

    def _check_magnetic(self, entry: dict) -> bool:
        if not self.config.magnetic_ordering: return True
        return entry.get("magnetic_ordering") in self.config.magnetic_ordering

    def _check_lattice(self, entry: dict) -> bool:
        if not self.config.exclude_lattice_types: return True
        return entry["bravais"]["lattice"] not in self.config.exclude_lattice_types

    def _check_centering(self, entry: dict) -> bool:
        if not self.config.exclude_centering_types: return True
        return entry["bravais"]["center"] not in self.config.exclude_centering_types

    def _check_bandgap(self, entry: dict) -> bool:
        bg_info = entry["bandgap_info"]
        method = self.config.bandgap_method
        
        # Check Value
        val = bg_info.get(f"bandgap_{method}")
        if val is None or not (self.config.bandgap_min <= val <= self.config.bandgap_max):
            return False
            
        # Check Type (Direct/Indirect)
        if self.config.gap_type:
            g_type = bg_info.get(f"gap_type_{method}")
            if g_type not in self.config.gap_type:
                return False
                
        return True

    def _check_size(self, entry: dict) -> bool:
        n = entry["orig"]["n_atoms"]
        if self.config.n_atoms_min is not None and n < self.config.n_atoms_min: return False
        if self.config.n_atoms_max is not None and n > self.config.n_atoms_max: return False
        return True

    def _check_soc(self, entry: dict) -> bool:
        if self.config.soc is None: return True
        return str(entry.get("soc")) == str(self.config.soc)

    def filter_dataset(self) -> List[dict]:
        """Apply all filters and return filtered entries."""
        checks = [
            self._check_magnetic,
            self._check_lattice,
            self._check_centering,
            self._check_bandgap,
            self._check_size,
            self._check_soc
        ]

        filtered = [entry for entry in self.data if all(f(entry) for f in checks)]
        return filtered

    def save_results(self, output_dir: Path, prefix: str = "f") -> Path:
        output_dir = Path(output_dir)
        filtered_data = self.filter_dataset()
        
        # Sort by atom count
        filtered_data.sort(key=lambda x: x["orig"]["n_atoms"])

        if self.config.keep_k == -1:
            self.config.keep_k = len(filtered_data)

        # Truncate to top K if requested
        if (self.config.keep_k is not None) and len(filtered_data) > self.config.keep_k:
            print(f"Limiting results to top {self.config.keep_k} entries (sorted by size).")
            filtered_data = filtered_data[:self.config.keep_k]
        
        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_dir = output_dir / f"{prefix}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = save_dir / "data.json"
        with open(data_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)

        log_data = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "database_source": str(self.database_path.absolute()),
            },
            "stats": {
                "total_entries": len(self.data),
                "filtered_entries": len(filtered_data),
                "retention_rate_percent": round((len(filtered_data) / len(self.data)) * 100, 2) if self.data else 0
            },
            "configuration": self.config.to_dict()
        }
        
        log_path = save_dir / "log.yaml"
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, sort_keys=False)

        print(f"\n--- Filtering Complete ---")
        print(f"Total entries:    {len(self.data)}")
        print(f"Kept entries:     {len(filtered_data)}")
        print(f"Directory created: {save_dir}")
        print(f"--------------------------\n")
        
        return save_dir

def main():
    parser = argparse.ArgumentParser(description="CIF filter")
    parser.add_argument("config", type=Path, help="Path to the JSON configuration file")
    parser.add_argument("--database", "-i", type=Path, default="data/cifs/all_data.json",
                        help="Path to the source database JSON")
    parser.add_argument("--output", "-o", type=Path, default="data/cifs/filtered",
                        help="Directory where results will be saved")
    parser.add_argument("--prefix", "-pre", type=str, default="run",
                        help="Prefix for the output folder name")
    
    args = parser.parse_args()
    
    config = FilterConfig.from_json(args.config)
    processor = DatasetFilter(args.database, config)
    processor.save_results(args.output, prefix=args.prefix)

if __name__ == "__main__":
    main()
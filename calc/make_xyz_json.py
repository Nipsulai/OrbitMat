#!/usr/bin/env python3
"""
Create input JSON for the CP2K molecular calculations (XYZ files)

Usage:
    python calc/make_xyz_json.py \\
        --xyz_dir data/xyzfiles \\
        --csv data/xyz/molecules.csv \\
        --label M06-2X \\
        --out xyz_input.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path
import random

import pandas as pd

EV_TO_HARTREE = 0.0367493

def main():
    parser = argparse.ArgumentParser(
        description="Generate xyz input JSON for CP2K workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xyz_dir", required=True, help="Directory containing XYZ files (dsgdb9nsd_XXXXXX.xyz)")
    parser.add_argument("--csv", required=True, help="CSV with DFT energies; must have an 'index' column")
    parser.add_argument("--label", required=True, help="CSV column with DFT energy in eV (e.g. M06-2X)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--index_col", default="index", help="Name of the index column in the CSV")
    parser.add_argument("--limit", type=int, default=None, help="Max number of entries to include")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling the XYZ files")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.set_index(args.index_col, inplace=True)
    df.index = df.index.astype(int)

    if args.label not in df.columns:
        print(f"Error: column '{args.label}' not found in CSV. Available: {list(df.columns)}")
        sys.exit(1)

    xyz_files = sorted(glob.glob(str(Path(args.xyz_dir) / "dsgdb9nsd_*.xyz")))
    if not xyz_files:
        print(f"No XYZ files found in {args.xyz_dir}")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(xyz_files)

    entries = []
    skipped_no_csv = 0
    skipped_nan = 0

    for xyz_path in xyz_files:
        stem = Path(xyz_path).stem  # e.g. dsgdb9nsd_000001
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            skipped_no_csv += 1
            continue

        if idx not in df.index:
            skipped_no_csv += 1
            continue

        energy_eV = df.loc[idx, args.label]
        if pd.isna(energy_eV):
            skipped_nan += 1
            continue

        entries.append({
            "cif_path": str(Path(xyz_path).resolve()),
            "energy_dft_Ha": float(energy_eV) * EV_TO_HARTREE,
        })

        if args.limit is not None and len(entries) >= args.limit:
            break

    print(f"Entries: {len(entries)}  |  skipped (not in CSV): {skipped_no_csv}  |  skipped (NaN energy): {skipped_nan}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

#Z_skip = {58, 45, 44, 42, 27, 52, 53, 28, 26, 24, 23, 25}
Z_skip = {58} #He, Ne, Ce 2,10

def filter_hdf5(src_path: str, dst_path: str) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        totals = {}
        for split in ("train", "val", "test"):
            if split not in src:
                continue
            split_grp = src[split]
            dst_split = dst.create_group(split)

            kept, skipped = 0, 0
            for mat_id in split_grp:
                atomic_numbers = np.asarray(split_grp[mat_id]["geom0"]["atomic_numbers"])
                if np.isin(atomic_numbers, list(Z_skip)).any():
                    skipped += 1
                    continue
                src.copy(split_grp[mat_id], dst_split, name=mat_id)
                kept += 1

            totals[split] = kept
            print(f"[{split:5s}] kept {kept}, removed {skipped}")

        print(f"\nFinal split sizes — train: {totals.get('train', 0)}, "
              f"val: {totals.get('val', 0)}, test: {totals.get('test', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a filtered copy.",
    )
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("--out", "-o", help="Output HDF5 file")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    dst = args.out or str(src.with_stem(f"{src.stem}_noCe"))

    print(f"Filtering {src} -> {dst}")
    filter_hdf5(str(src), dst)
    print("Done.")

if __name__ == "__main__":
    main()

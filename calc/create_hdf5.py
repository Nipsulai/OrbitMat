#!/usr/bin/env python3
"""
Create an HDF5 dataset from CP2K output directories.

Usage:
    # New splits
    python calc/create_hdf5.py calc/out/tzvp_run --train 6000 --val 800 --test 800 \
        --method tzvp --out calc/tzvp.hdf5

    # Reuse splits from existing HDF5
    python calc/create_hdf5.py calc/out/tzvp_run --from_hdf5 prev.hdf5 --method tzvp

    # Merge two folders
    python calc/create_hdf5.py calc/out/run1 calc/out/run2 --train 6000 \
        --method tzvp --out calc/combined.hdf5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import METHODS
from src.hdf5_utils import get_npz_paths, make_splits, write_split
from src.postproc import norb_by_z


def create_hdf5(
    cp2k_folders: list[Path],
    output_path: Path,
    method: str,
    train_size: int = 0,
    val_size: int = 0,
    test_size: int = 0,
    seed: int = 42,
    topk: int = 32,
    n_workers: int = 1,
    from_hdf5: Path | None = None,
    use_dist: bool = False,
) -> None:
    output_path = Path(output_path).absolute()
    if from_hdf5 is None:
        output_path = output_path.with_stem(f"{output_path.stem}_s{seed}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect available NPZ files across all folders
    id_to_npz: dict = {}
    for folder in cp2k_folders:
        folder = Path(folder).absolute()
        folder_map = get_npz_paths(folder)
        overlap = set(folder_map) & set(id_to_npz)
        if overlap:
            print(f"[WARN] {len(overlap)} duplicate IDs from {folder.name} will overwrite previous entries.")
        id_to_npz.update(folder_map)
        print(f"  {folder.name}: {len(folder_map)} calculations found.")

    if not id_to_npz:
        print("No successful calculations found.")
        sys.exit(1)
    print(f"Total: {len(id_to_npz)} across {len(cp2k_folders)} folder(s).")

    # Build or load splits
    if from_hdf5 is not None:
        from_hdf5 = Path(from_hdf5)
        if not from_hdf5.exists():
            print(f"Source HDF5 not found: {from_hdf5}")
            sys.exit(1)
        print(f"Reading splits from {from_hdf5.name}...")
        splits_dict: dict = {}
        with h5py.File(from_hdf5, "r") as src:
            for split in ("train", "val", "test"):
                if split in src:
                    ids = list(src[split].keys())
                    splits_dict[split] = np.array(ids)
                    print(f"  {split}: {len(ids)}")
                else:
                    splits_dict[split] = np.array([])
    else:
        print(f"Creating splits (seed={seed})...")
        splits = make_splits(
            np.array(list(id_to_npz.keys())),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        splits_dict = splits.as_dict()
        print(f"  train: {len(splits.train)}, val: {len(splits.val)}, test: {len(splits.test)}")

    mcfg = METHODS[method]
    norb_z = norb_by_z(method) if mcfg.norb_json_path is not None else None

    with h5py.File(output_path, "w") as h5:
        for split_name, split_ids in splits_dict.items():
            if len(split_ids) == 0:
                continue
            present = [mid for mid in split_ids if mid in id_to_npz]
            missing = len(split_ids) - len(present)
            if missing:
                print(f"[WARN] {split_name}: {missing} IDs not found in folders.")
            write_split(
                h5_file=h5,
                split_name=split_name,
                material_ids=split_ids,
                id_to_npz=id_to_npz,
                method=method,
                norb_z=norb_z,
                topk=topk,
                n_workers=n_workers,
                use_dist=use_dist,
            )

    print(f"Done. HDF5 saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create HDF5 dataset from CP2K output directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", nargs="+", help="CP2K output folder(s).")
    parser.add_argument("--out", "-o", default=None, help="Output HDF5 file path.")
    parser.add_argument("--method", "-m", required=True, choices=list(METHODS),
                        help="DFT method.")
    parser.add_argument("--train",  type=int, default=0)
    parser.add_argument("--val",    type=int, default=0)
    parser.add_argument("--test",   type=int, default=0)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--topk",   type=int, default=32,
                        help="Max neighbor blocks per atom (periodic only).")
    parser.add_argument("--distance", action="store_true",
                        help="Select neighbors by distance instead of matrix-norm score.")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--from_hdf5", default=None,
                        help="Existing HDF5 to read splits from (overrides --train/val/test).")
    args = parser.parse_args()

    if args.from_hdf5 is None and args.train == 0 and args.val == 0 and args.test == 0:
        parser.error("Specify at least one of --train, --val, --test.")

    if args.out is None:
        name = "_".join(Path(f).name for f in args.folder)
        args.out = f"calc/{name}.hdf5"

    create_hdf5(
        cp2k_folders=args.folder,
        output_path=args.out,
        method=args.method,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
        seed=args.seed,
        topk=args.topk,
        n_workers=args.workers,
        from_hdf5=args.from_hdf5,
        use_dist=args.distance,
    )


if __name__ == "__main__":
    main()

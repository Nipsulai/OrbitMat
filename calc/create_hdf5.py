#!/usr/bin/env python3
"""
Postprocessing script to create HDF5 dataset from computed NPZ/pickle files.

Usage:
    # Mode 1: Create new random splits
    python calc/create_hdf5.py {path_to_folder} --train 800 --val 100 --test 100 --method pbe

    # Mode 2: Reuse splits from an existing HDF5 file
    python calc/create_hdf5.py {path_to_folder} --from_hdf5 previous_data.hdf5 --method pbe

This script:
1. Scans all folders in the specified output directory
2. Loads pickle files from successful calculations
3. Writes everything to a single HDF5 file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hdf5_utils import (
    get_npz_paths,
    make_splits,
    write_pbc_split,
)
from src.postproc import norb_by_z


def create_pbc_hdf5(
    cp2k_folder: Path,
    output_path: Path,
    method: str,
    train_size: int = 0,
    val_size: int = 0,
    test_size: int = 0,
    seed: int = 42,
    topk: int = 32,
    n_workers: int = 1,
    from_hdf5: Path | None = None,
):
    cp2k_folder = Path(cp2k_folder).absolute()
    output_path = Path(output_path).absolute()
    
    # If generating new splits, append seed to filename. 
    # If copying, keep name or append 'reprocessed' to distinguish? 
    # Here we stick to the seed convention if not copying, or just use output_path as is.
    if from_hdf5 is None:
        output_path = output_path.with_stem(f"{output_path.stem}_s{seed}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Scan folder for calculation results
    id_to_npz = get_npz_paths(cp2k_folder, npz=False)

    if not id_to_npz:
        print(f"[ERROR] No successful calculations found in {cp2k_folder}")
        sys.exit(1)

    print(f"[INFO] Found {len(id_to_npz)} successful calculations in folder.")

    # 2. Determine Splits
    splits_dict = {}
    
    if from_hdf5 is not None:
        from_hdf5 = Path(from_hdf5)
        if not from_hdf5.exists():
            print(f"[ERROR] Source HDF5 file not found: {from_hdf5}")
            sys.exit(1)
            
        print(f"[INFO] Reading split definitions from {from_hdf5}...")
        with h5py.File(from_hdf5, "r") as f_src:
            for split_name in ["train", "val", "test"]:
                if split_name in f_src:
                    # The keys in the group are the material IDs
                    ids = list(f_src[split_name].keys())
                    splits_dict[split_name] = np.array(ids)
                    print(f"       - {split_name}: {len(ids)} IDs loaded")
                else:
                    splits_dict[split_name] = np.array([])
    else:
        print(f"[INFO] Generating new random splits (Seed: {seed})...")
        available_ids = np.array(list(id_to_npz.keys()))
        splits = make_splits(
            available_ids,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        splits_dict = splits.as_dict()
        print(f"       - train: {len(splits.train)}")
        print(f"       - val:   {len(splits.val)}")
        print(f"       - test:  {len(splits.test)}")

    # 3. Process and Write
    orb_map = norb_by_z(method)

    with h5py.File(output_path, "w") as h5:
        # Copy metadata if from_hdf5? (Optional, skipping for now to keep it clean)
        
        for split_name, split_ids in splits_dict.items():
            if len(split_ids) == 0:
                continue

            # Calculate how many IDs from the split actually exist in our folders
            present_ids = [mid for mid in split_ids if mid in id_to_npz]
            if len(present_ids) < len(split_ids):
                print(f"[WARN] Split '{split_name}': {len(split_ids)} IDs in source, but only {len(present_ids)} found in folder.")

            write_pbc_split(
                h5_file=h5,
                split_name=split_name,
                material_ids=split_ids,
                id_to_npz=id_to_npz,
                npz=False,
                norb_by_z=orb_map,
                method=method,
                topk=topk,
                n_workers=n_workers,
            )
            
    print(f"Done. HDF5 saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create HDF5 dataset from computed NPZ/pickle files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the CP2K output folder",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--from_hdf5",
        type=str,
        default=None,
        help="Path to an existing HDF5 file to copy train/val/test splits from. If provided, overrides --train/--val/--test.",
    )
    # Made optional (required=False) to support --from_hdf5 mode
    parser.add_argument(
        "--train",
        type=int,
        default=0,
        help="Number of samples for training (Required if --from_hdf5 is not set)",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=0,
        help="Number of samples for validation (Required if --from_hdf5 is not set)",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="Number of samples for testing (Required if --from_hdf5 is not set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["pbe", "xtb"],
        required=True,
        help="DFT method used (pbe or xtb)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=32,
        help="Max neighbor blocks per atom",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers for loading data",
    )

    args = parser.parse_args()

    # Manual validation for required args depending on mode
    if args.from_hdf5 is None:
        if args.train == 0 and args.val == 0 and args.test == 0:
             parser.error("When --from_hdf5 is not used, you must specify at least one of --train, --val, or --test.")

    if args.out is None:
        args.out = "calc/" + Path(args.folder).name + ".hdf5"

    create_pbc_hdf5(
        cp2k_folder=args.folder,
        output_path=args.out,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
        method=args.method,
        seed=args.seed,
        topk=args.topk,
        n_workers=args.workers,
        from_hdf5=args.from_hdf5,
    )


if __name__ == "__main__":
    main()
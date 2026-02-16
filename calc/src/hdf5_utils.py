from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import h5py
import pickle

from scipy.sparse import issparse
from .perm_and_blocks import _permute_block, _permute_block_pbe
from collections import defaultdict

### General methods ###

from scipy.sparse.linalg import norm as sp_norm

def mat_norm(m):
    return sp_norm(m) if issparse(m) else np.linalg.norm(m)

@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

def get_npz_paths(run_dir: Path, npz=False) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    npz_paths: Dict[str, Path] = {}

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue

        #meta_file = subdir / "metadata.json"
        #if not meta_file.exists():
        #    continue

        #with meta_file.open("r") as f:
        #    meta = json.load(f)

        #if not meta.get("success", False):
        #    continue
        if npz:
            file_path = subdir / "matrices.npz"
        else:
            file_path = subdir / "matrices.pkl"
        if file_path.exists():
            npz_paths[subdir.name] = file_path
        else:
            continue

    return npz_paths


def make_splits(
    available: np.ndarray,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> SplitIndices:
    rng = np.random.default_rng(seed)
    arr = np.array(available, copy=True)
    rng.shuffle(arr)

    n_total = len(arr)
    n_train = min(train_size, n_total)
    n_val = min(val_size, max(0, n_total - n_train))
    n_test = min(test_size, max(0, n_total - n_train - n_val))

    train = arr[:n_train]
    val = arr[n_train : n_train + n_val]
    test = arr[n_train + n_val : n_train + n_val + n_test]
    return SplitIndices(train=train, val=val, test=test)

### Methods for PBC materials ###

def parse_cp2k_charges(out_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parses Mulliken and Hirshfeld charges from a CP2K output file."""
    if not out_file.exists():
        return None, None

    mulliken, hirshfeld = [], []
    in_mulliken, in_hirshfeld = False, False

    with open(out_file, "r") as f:
        for line in f:
            clean = line.strip()
            if not clean: continue

            # Detect sections
            if "Mulliken Population Analysis" in line:
                in_mulliken, mulliken = True, []
                continue
            if "Hirshfeld Charges" in line:
                in_hirshfeld, hirshfeld = True, []
                continue

            # Parse Mulliken
            if in_mulliken:
                parts = clean.split()
                if parts[0].isdigit():
                    mulliken.append(float(parts[-1]))
                elif "Total charge" in line or "!" in line:
                    in_mulliken = False
            
            # Parse Hirshfeld
            if in_hirshfeld:
                parts = clean.split()
                if parts[0].isdigit():
                    hirshfeld.append(float(parts[-1]))
                elif "Total Charge" in line or "!" in line:
                    in_hirshfeld = False

    m_arr = np.array(mulliken) if mulliken else None
    h_arr = np.array(hirshfeld) if hirshfeld else None
    return m_arr, h_arr

def convert_matrices_to_blocks(data: Dict, norb_by_z: Dict[int, int], method: str = "xtb", topk: int = 32, cutoff: float = 1e-32) -> list:
    """
    Convert matrices (per translation T) to blocks.

    Args:
        data: Dict with 'matrices', 'atomic_numbers', 'geometry_bohr', 'cell_bohr'
        norb_by_z: Dict mapping atomic number to number of orbitals (basis functions)
        method: 'xtb' or 'pbe' to determine which permutation to apply
        topk: Max neighbor blocks per atom
        cutoff: Score threshold for including blocks

    Returns:
        List of block dicts
    """
    if method == "pbe":
        permute_block = _permute_block_pbe
    else:
        permute_block = _permute_block

    T_matrices = data["matrices"]
    elem_numbers = data["atomic_numbers"]
    coords_bohr = data["geometry_bohr"]
    cell_bohr = data["cell_bohr"]
    n_atoms = len(elem_numbers)

    # Build atom orbital ranges
    atom_ranges = []
    offset = 0
    for z in elem_numbers:
        norb = norb_by_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    def get_block(mat, src, ngb):
        """Extract block for atoms src, ngb (1-indexed)."""
        r_start, r_end = atom_ranges[src - 1]
        c_start, c_end = atom_ranges[ngb - 1]
        if issparse(mat):
            return mat[r_start:r_end, c_start:c_end].toarray()
        return mat[r_start:r_end, c_start:c_end]

    def compute_block_norm(mat, src, ngb):
        block = get_block(mat, src, ngb)
        return float(np.sum(block ** 2))

    blocks_by_source = defaultdict(list)

    # Precompute norms per T-vector
    #T_norms = {}
    #for T_vec, mats in T_matrices.items():
    #    T_norms[T_vec] = {k: mat_norm(mats[k]) for k in ('H', 'S', 'F', 'P')}

    for T_vec, mats in T_matrices.items():
        F_T, P_T, S_T, H_T = mats["F"], mats["P"], mats["S"], mats["H"]

        for src in range(1, n_atoms + 1):
            for ngb in range(1, n_atoms + 1):
                nF = compute_block_norm(F_T, src, ngb)
                nS = compute_block_norm(S_T, src, ngb)
                nH = compute_block_norm(H_T, src, ngb)
                score = nF * nS * nH

                if score < cutoff:
                    continue

                blocks_by_source[src].append({
                    'ngb': ngb,
                    'cell': T_vec,
                    'T_vec': T_vec,
                    'score': score,
                    'mats': mats,
                })

    # Select top-k
    blocks = []
    self_ctr, pair_ctr = 0, 0
    a1, a2, a3 = cell_bohr[0], cell_bohr[1], cell_bohr[2]

    for src in range(1, n_atoms + 1):
        atom_blocks = blocks_by_source.get(src, [])

        self_block = None
        neighbor_blocks = []

        for block in atom_blocks:
            if block['ngb'] == src and block['cell'] == (0, 0, 0):
                self_block = block
            else:
                neighbor_blocks.append(block)

        # Add self-block
        if self_block is not None:
            mats = self_block['mats']
            #norms = T_norms[self_block['T_vec']]
            ngb = self_block['ngb']
            z_src = int(elem_numbers[src - 1])
            z_ngb = int(elem_numbers[ngb - 1])

            Hb = permute_block(get_block(mats['H'], src, ngb), z_src, z_ngb)
            Fb = permute_block(get_block(mats['F'], src, ngb), z_src, z_ngb)

            blocks.append({
                "is_self": True,
                "ctr": self_ctr,
                "source": src,
                "neighbor": ngb,
                "cell": self_block['cell'],
                "matrix": {
                    'H': Hb,
                    'S': permute_block(get_block(mats['S'], src, ngb), z_src, z_ngb),
                    'F': Fb,
                    'P': permute_block(get_block(mats['P'], src, ngb), z_src, z_ngb),
                    'F_H': Fb-Hb,
                    'score': self_block['score'],
                },
            })
            self_ctr += 1

        # Top-k neighbor blocks
        neighbor_blocks.sort(key=lambda b: b['score'], reverse=True)
        selected = neighbor_blocks[:topk]

        for block in selected:
            ngb = block['ngb']
            mats = block['mats']
            #norms = T_norms[block['T_vec']]

            ic1, ic2, ic3 = block['cell']
            z_src = int(elem_numbers[src - 1])
            z_ngb = int(elem_numbers[ngb - 1])

            # Calculate distance
            r_i = coords_bohr[src - 1]
            r_j = coords_bohr[ngb - 1]
            shift = ic1 * a1 + ic2 * a2 + ic3 * a3
            dist = float(np.linalg.norm(r_j - r_i + shift))

            Hb = permute_block(get_block(mats['H'], src, ngb), z_src, z_ngb)
            Fb = permute_block(get_block(mats['F'], src, ngb), z_src, z_ngb)

            blocks.append({
                "is_self": False,
                "ctr": pair_ctr,
                "source": src,
                "neighbor": ngb,
                "cell": block['cell'],
                "matrix": {
                    'H': Hb,
                    'S': permute_block(get_block(mats['S'], src, ngb), z_src, z_ngb),
                    'F': Fb,
                    'P': permute_block(get_block(mats['P'], src, ngb), z_src, z_ngb),
                    'F_H': Fb-Hb,
                    'score': block['score'],
                },
                "dist": dist,
            })
            pair_ctr += 1

    return blocks


def load_npz_pbc(npz_path: Path, npz, norb_by_z: Dict[int, int] = None, method: str = "xtb", topk: int = 32) -> Dict[str, object]:
    """Load NPZ or pickle file for periodic (PBC) materials with block-based matrices.
    """
    if npz:
        with np.load(npz_path, allow_pickle=True) as z:
            element_numbers = z["atomic_numbers"]
            coords_bohr = z["geometry_bohr"]
            net_spin = int(z["net_spin"]) if "net_spin" in z.files else 0
            charge = int(z["charge"])
            cell_bohr = z["cell_bohr"]
            pbc = z["pbc"]
            energy_xtb = float(z["energy_xtb_Ha"]) if "energy_xtb_Ha" in z.files else None

            blocks = z["blocks"]

            # Optional bandgap info
            bandgap_pbe = float(z["bandgap_pbe"]) if "bandgap_pbe" in z.files else None
            bandgap_hse = float(z["bandgap_hse"]) if "bandgap_hse" in z.files else None
            bandgap_cp2k = float(z["bandgap_cp2k"]) if "bandgap_cp2k" in z.files else None
            gap_type_pbe = str(z["gap_type_pbe"]) if "gap_type_pbe" in z.files else None
            gap_type_hse = str(z["gap_type_hse"]) if "gap_type_hse" in z.files else None
    else:
        # Load pickle file from postproc_matrices
        with open(npz_path, "rb") as f:
            data = pickle.load(f)

        element_numbers = data["atomic_numbers"]
        coords_bohr = data["geometry_bohr"]
        net_spin = int(data.get("net_spin", 0))
        charge = int(data.get("charge", 0))
        cell_bohr = data["cell_bohr"]
        pbc = data["pbc"]
        energy_xtb = float(data["energy_cp2k_Ha"]) if "energy_cp2k_Ha" in data else None

        # Convert T-matrices to blocks format
        if norb_by_z is None:
            raise ValueError("norb_by_z is required when loading pickle files")
        blocks = convert_matrices_to_blocks(data, norb_by_z, method=method, topk=topk)

        # Optional bandgap info
        bandgap_pbe = float(data["bandgap_pbe"]) if "bandgap_pbe" in data else None
        bandgap_hse = float(data["bandgap_hse"]) if "bandgap_hse" in data else None
        bandgap_cp2k = float(data.get("bandgap_cp2k")) if "bandgap_cp2k" in data else None
        gap_type_pbe = str(data["gap_type_pbe"]) if "gap_type_pbe" in data else None
        gap_type_hse = str(data["gap_type_hse"]) if "gap_type_hse" in data else None

        out_cp2k_path = npz_path.parent / "out.cp2k"
        m_charges, h_charges = parse_cp2k_charges(out_cp2k_path)

        n_atoms = len(element_numbers)
        if m_charges is not None and len(m_charges) != n_atoms:
            print(f"[WARN] Mulliken charge count mismatch in {out_cp2k_path}")
            m_charges = None
        if h_charges is not None and len(h_charges) != n_atoms:
            print(f"[WARN] Hirshfeld charge count mismatch in {out_cp2k_path}")
            h_charges = None


    return {
        "atomic_numbers": element_numbers,
        "geometry_bohr": coords_bohr,
        "charge": charge,
        "net_spin": net_spin,
        "energy_xtb_Ha": energy_xtb,
        "cell_bohr": cell_bohr,
        "pbc": pbc,
        "blocks": blocks,
        "bandgap_pbe": bandgap_pbe,
        "bandgap_hse": bandgap_hse,
        "bandgap_cp2k": bandgap_cp2k,
        "gap_type_pbe": gap_type_pbe,
        "gap_type_hse": gap_type_hse,
        "m_charges": m_charges,
        "h_charges": h_charges,
    }


def write_pbc_material(
    h5_group: h5py.Group,
    material_id: str,
    payload: Dict[str, object],
) -> None:
    """
    Write a single PBC material to an HDF5 group.

    Structure:
        {material_id}/
            geom0/
                atomic_numbers
                geometry_bohr
                charge
                net_spin
                cell_bohr
                pbc
                energy_xtb_Ha (optional)
                bandgap_pbe_eV (optional)
                bandgap_hse_eV (optional)
                gap_type_pbe (optional)
                gap_type_hse (optional)
                self_idx    - index array for self blocks
                pair_idx    - index array for pair blocks
                self_{i}/2body/H,S,F,P
                pair_{i}/2body/H,S,F,P
    """
    grp_mat = h5_group.create_group(material_id)
    geom0 = grp_mat.create_group("geom0")

    # Write geometry info
    geom0.create_dataset("atomic_numbers", data=payload["atomic_numbers"], dtype="i8")
    geom0.create_dataset("geometry_bohr", data=payload["geometry_bohr"], dtype="f8")
    geom0.create_dataset("charge", data=payload["charge"], dtype="f8")
    geom0.create_dataset("m_charges", data=payload["m_charges"], dtype="f8")
    geom0.create_dataset("h_charges", data=payload["h_charges"], dtype="f8")
    geom0.create_dataset("net_spin", data=payload["net_spin"], dtype="f8")
    geom0.create_dataset("cell_bohr", data=payload["cell_bohr"], dtype="f8")
    geom0.create_dataset("pbc", data=payload["pbc"], dtype="i8")

    if payload["energy_xtb_Ha"] is not None:
        geom0.create_dataset("energy_xtb_Ha", data=payload["energy_xtb_Ha"], dtype="f8")

    if payload["bandgap_pbe"] is not None:
        geom0.create_dataset("bandgap_pbe_eV", data=payload["bandgap_pbe"], dtype="f8")
    if payload["bandgap_hse"] is not None:
        geom0.create_dataset("bandgap_hse_eV", data=payload["bandgap_hse"], dtype="f8")
    if payload["bandgap_cp2k"] is not None:
        geom0.create_dataset("bandgap_cp2k_eV", data=payload["bandgap_cp2k"], dtype="f8")
    if payload["gap_type_pbe"] is not None:
        geom0.attrs["gap_type_pbe"] = payload["gap_type_pbe"]
    if payload["gap_type_hse"] is not None:
        geom0.attrs["gap_type_hse"] = payload["gap_type_hse"]

    # Process blocks
    blocks = payload["blocks"]
    zs = payload["atomic_numbers"]
    coords = payload["geometry_bohr"]
    cell = payload["cell_bohr"]
    a1, a2, a3 = cell[0], cell[1], cell[2]

    self_rows = []
    pair_rows = []

    for block in blocks:
        is_self = block["is_self"]
        ctr = block["ctr"]
        src_idx = int(block["source"]) - 1  # Convert to 0-indexed
        nbr_idx = int(block["neighbor"]) - 1
        ic1, ic2, ic3 = block["cell"]
        mat = block["matrix"]

        if is_self:
            name = f"self_{int(ctr)}"
            self_rows.append([ctr, src_idx, nbr_idx, int(ic1), int(ic2), int(ic3)])
        else:
            name = f"pair_{int(ctr)}"
            dist = block.get("dist", 0.0)
            score = mat.get("score", 0.0)
            pair_rows.append([ctr, src_idx, nbr_idx, int(ic1), int(ic2), int(ic3), score, dist])

        p = geom0.create_group(name)
        p0 = p.create_group("2body")
        p0.create_dataset("H", data=mat["H"], dtype="f8")
        p0.create_dataset("S", data=mat["S"], dtype="f8")
        p0.create_dataset("F", data=mat["F"], dtype="f8")
        p0.create_dataset("P", data=mat["P"], dtype="f8")
        p0.create_dataset("F_H", data=mat["F_H"], dtype="f8")
        #REMEMBER TO CHECK ONE-BODY AND SYMMETRICITY

    # Write index arrays
    if self_rows:
        geom0.create_dataset("self_idx", data=np.array(self_rows, dtype="int64"), dtype="i8")
    if pair_rows:
        pair_arr = np.array([tuple(row) for row in pair_rows], dtype="f8")
        geom0.create_dataset("pair_idx", data=pair_arr)


def _load_material_worker(args: Tuple) -> Tuple[str, Optional[Dict]]:
    """
    Worker function for parallel loading of materials.

    Args:
        args: Tuple of (mat_id, npz_path, npz, norb_by_z, method, topk)

    Returns:
        Tuple of (mat_id, payload) or (mat_id, None) if loading fails
    """
    mat_id, npz_path, npz, norb_by_z, method, topk = args
    payload = load_npz_pbc(npz_path, npz, norb_by_z=norb_by_z, method=method, topk=topk)
    return (mat_id, payload)


def write_pbc_split(
    h5_file: h5py.File,
    split_name: str,
    material_ids: Iterable[str],
    id_to_npz: Dict[str, Path],
    npz: bool = False,
    norb_by_z: Dict[int, int] = None,
    method: str = "xtb",
    topk: int = 32,
    n_workers: int = 1,
) -> int:
    """
    Write a train/val/test split for PBC materials.
    """
    split_grp = h5_file.create_group(split_name)
    count = 0

    # Filter to valid material IDs
    valid_ids = [mid for mid in material_ids if mid in id_to_npz]

    if n_workers > 1:
        # Parallel loading with sequential writes
        args_list = [
            (mat_id, id_to_npz[mat_id], npz, norb_by_z, method, topk)
            for mat_id in valid_ids
        ]

        with Pool(n_workers) as pool:
            results = pool.imap(_load_material_worker, args_list)
            for mat_id, payload in tqdm(results, total=len(args_list), desc=f"[{split_name}]"):
                if payload is not None:
                    write_pbc_material(split_grp, mat_id, payload)
                    count += 1
    else:
        # Sequential loading and writing (original behavior)
        for mat_id in tqdm(valid_ids, desc=f"[{split_name}]"):
            npz_path = id_to_npz.get(mat_id)
            if npz_path is None:
                continue

            payload = load_npz_pbc(npz_path, npz, norb_by_z=norb_by_z, method=method, topk=topk)
            write_pbc_material(split_grp, mat_id, payload)
            count += 1

    return count
    

### METHODS for QM9 molecules (XYZ) ###

def mol_name_from_idx(idx: int) -> str:
    return f"dsgdb9nsd_{idx:06d}"

def load_npz_payload(npz_path: Path) -> Dict[str, object]:
    """
    Loads required keys. Will raise KeyError if missing.
    """
    with np.load(npz_path) as z:
        element_numbers = z["atomic_numbers"]
        coords_bohr = z["geometry_bohr"]

        charge_arr = z["charge"]
        charge = int(charge_arr) if np.ndim(charge_arr) == 0 else int(charge_arr[()])

        energy_xtb = z["energy_xtb_Ha"] if "energy_xtb_Ha" in z.files else None

        F = z["F"]
        P = z["P"]
        S = z["S"]
        H = z["H"]

    return {
        "atomic_numbers": element_numbers,
        "geometry_bohr": coords_bohr,
        "charge": charge,
        "energy_xtb_Ha": energy_xtb,
        "F": F,
        "P": P,
        "S": S,
        "H": H,
    }


def load_molecules_csv(csv_file, index_col="index"):
    df = pd.read_csv(str(csv_file))
    df.set_index(index_col, inplace=True)
    df.index = df.index.astype(int)
    return df

def build_idx_to_npz(run_dir: Path) -> Dict[int, Path]:
    """
    Convenience wrapper for molecule datasets (dsgdb9nsd_<idx>).
    Maps integer index -> matrices.npz path.
    """
    npz_paths = get_npz_paths(run_dir)
    idx_to_npz: Dict[int, Path] = {}

    for name, npz_path in npz_paths.items():
        if name.startswith("dsgdb9nsd_"):
            idx = int(name.split("_")[-1])
            idx_to_npz[idx] = npz_path

    return idx_to_npz


def compute_available_indices(
    idx_to_npz: Dict[int, Path],
    df: pd.DataFrame,
    dft_label: str,
) -> np.ndarray:
    """
    success ∩ CSV ∩ non-NaN dft_label
    """
    available = []
    for idx in idx_to_npz.keys():
        if idx not in df.index:
            continue
        if pd.isna(df.loc[idx, dft_label]):
            continue
        available.append(idx)

    return np.array(sorted(available), dtype=int)

def write_one_molecule_pair(
    split_full: h5py.Group,
    split_tbl: h5py.Group,
    mol_name: str,
    payload: Dict[str, object],
    net_spin: int,
    dft_energy_Ha: float,
    dft_label: str,
) -> None:
    """
    Writes:
      - full: atomic_numbers, geometry_bohr, charge, net_spin, optional energy_xtb_Ha,
              2body(F,P,S,H), {dft_label}_energy_Ha
      - tbl : atomic_numbers, geometry_bohr, charge, net_spin
    """
    # FULL
    grp_mol_full = split_full.create_group(mol_name)
    geo_full = grp_mol_full.create_group("0")

    geo_full.create_dataset("atomic_numbers", data=payload["atomic_numbers"])
    geo_full.create_dataset("geometry_bohr", data=payload["geometry_bohr"])
    if payload["energy_xtb_Ha"] is not None:
        geo_full.create_dataset("energy_xtb_Ha", data=payload["energy_xtb_Ha"])

    geo_full.create_dataset("charge", data=payload["charge"])
    geo_full.create_dataset("net_spin", data=net_spin)

    two_body = geo_full.create_group("2body")
    two_body.create_dataset("F", data=payload["F"])
    two_body.create_dataset("P", data=payload["P"])
    two_body.create_dataset("S", data=payload["S"])
    two_body.create_dataset("H", data=payload["H"])

    geo_full.create_dataset(f"{dft_label}_energy_Ha", data=dft_energy_Ha)

    # TBLITE (structure only)
    grp_mol_tbl = split_tbl.create_group(mol_name)
    geo_tbl = grp_mol_tbl.create_group("0")

    geo_tbl.create_dataset("atomic_numbers", data=payload["atomic_numbers"])
    geo_tbl.create_dataset("geometry_bohr", data=payload["geometry_bohr"])
    geo_tbl.create_dataset(f"{dft_label}_energy_Ha", data=dft_energy_Ha)
    geo_tbl.create_dataset("charge", data=payload["charge"])
    geo_tbl.create_dataset("net_spin", data=net_spin)

def write_split_pair(
    f_full: h5py.File,
    f_tbl: h5py.File,
    split_name: str,
    indices: Iterable[int],
    df: pd.DataFrame,
    idx_to_npz: Dict[int, Path],
    dft_label: str,
    ev_to_hartree: float,
    net_spin_default: int = 0,
) -> None:
    """
    Writes one split into both files. No skipping other than:
      - idx missing in CSV
      - NaN energy in CSV
      - missing matrices.npz path in idx_to_npz
    """

    split_full = f_full.create_group(split_name)
    split_tbl = f_tbl.create_group(split_name)

    for idx in tqdm(list(indices), desc=f"[{split_name}]"):
        idx = int(idx)

        if idx not in df.index:
            continue
        if pd.isna(df.loc[idx, dft_label]):
            continue

        npz_path = idx_to_npz.get(idx)
        if npz_path is None:
            continue

        payload = load_npz_payload(npz_path)

        mol_name = mol_name_from_idx(idx)
        dft_energy_Ha = float(df.loc[idx, dft_label]) * ev_to_hartree

        write_one_molecule_pair(
            split_full=split_full,
            split_tbl=split_tbl,
            mol_name=mol_name,
            payload=payload,
            net_spin=net_spin_default,
            dft_energy_Ha=dft_energy_Ha,
            dft_label=dft_label,
        )
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
from .perm_and_blocks import _permute_block, _permute_block_pbe, _permute_block_scan, _permute_block_tzvp, get_perm_map, apply_perm_map
from collections import defaultdict

### General methods ###

#python calc/create_hdf5.py calc/out/nmpbe calc/out/magnetic  --topk 32 --method pbe --train 6879 --val 850 --test 850 --out calc/pbe_fulldata_32.hdf5 --workers 30 

from scipy.sparse.linalg import norm as sp_norm

DIAG_BLOCKS = "/home/nikolai/OrbitMat/calc/src/input/elem_diag_mats.npy"

diag_blocks = np.load(DIAG_BLOCKS, allow_pickle=True).item()

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

def convert_matrices_to_blocks(data: Dict, norb_by_z: Dict[int, int], method: str = "xtb", topk: int = 32, cutoff: float = 1e-7, use_dist: bool = False) -> list:
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
    elif method == "scan":
        permute_block = _permute_block_scan
    elif method == "tzvp":
        permute_block = _permute_block_tzvp
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

    # Precompute orbital-to-atom mapping for vectorized block norm computation
    total_orbs = atom_ranges[-1][1]
    starts = np.array([rs for rs, re in atom_ranges], dtype=np.intp)
    orb_to_atom = np.empty(total_orbs, dtype=np.intp)
    for i, (rs, re) in enumerate(atom_ranges):
        orb_to_atom[rs:re] = i

    def _block_norms(mat):
        """Return (n_atoms, n_atoms) matrix of squared Frobenius block norms."""
        if issparse(mat):
            coo = mat.tocoo()
            norms = np.zeros((n_atoms, n_atoms))
            if coo.nnz:
                np.add.at(norms, (orb_to_atom[coo.row], orb_to_atom[coo.col]), coo.data ** 2)
        else:
            sq = np.asarray(mat) ** 2
            row_sums = np.add.reduceat(sq, starts, axis=0)
            norms = np.add.reduceat(row_sums, starts, axis=1)
        return norms

    blocks_by_source = defaultdict(list)
    a1, a2, a3 = cell_bohr[0], cell_bohr[1], cell_bohr[2]

    if use_dist:
        buffer_size = topk * 5
        for T_vec, mats in T_matrices.items():
            ic1, ic2, ic3 = T_vec
            shift = ic1 * a1 + ic2 * a2 + ic3 * a3
            # dists_T[src0, ngb0] = |r_ngb - r_src + shift|
            diffs = coords_bohr[np.newaxis, :, :] - coords_bohr[:, np.newaxis, :] + shift
            dists_T = np.linalg.norm(diffs, axis=2)
            for src0 in range(n_atoms):
                row = dists_T[src0].tolist()
                blocks_by_source[src0 + 1].extend([
                    {'ngb': ngb0 + 1, 'cell': T_vec, 'T_vec': T_vec,
                     '_dist': row[ngb0], 'mats': mats, 'score': 0.0}
                    for ngb0 in range(n_atoms)
                ])
            # Prune each source buffer to buffer_size after each T_vec
            for src_key in range(1, n_atoms + 1):
                lst = blocks_by_source[src_key]
                if len(lst) > buffer_size:
                    lst.sort(key=lambda b: b['_dist'])
                    blocks_by_source[src_key] = lst[:buffer_size]
    else:
        ctr = 0
        for T_vec, mats in T_matrices.items():
            scores = _block_norms(mats["F"])# * _block_norms(mats["P"]) # if you change this change cutoff
            src_arr, ngb_arr = np.where(scores >= cutoff)
            ctr += 1
            if ctr >= 120:
                break
            for src0, ngb0 in zip(src_arr.tolist(), ngb_arr.tolist()):
                blocks_by_source[src0 + 1].append({
                    'ngb': ngb0 + 1,
                    'cell': T_vec,
                    'T_vec': T_vec,
                    'score': float(scores[src0, ngb0]),
                    'mats': mats,
                })

    # Select top-k
    blocks = []
    self_ctr, pair_ctr = 0, 0
    max_nbr_selected = 0

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
            # Self-block: neighbor is always src itself.
            assert self_block['ngb'] == src
            z_src = int(elem_numbers[src - 1])

            H_blc = get_block(mats['H'], src, src)
            F_blc = get_block(mats['F'], src, src)
            S_blc = get_block(mats['S'], src, src)
            P_blc = get_block(mats['P'], src, src)

            Hb = permute_block(H_blc, z_src, z_src)
            Fb = permute_block(F_blc, z_src, z_src)
            Sb = permute_block(S_blc, z_src, z_src)
            Pb = permute_block(P_blc, z_src, z_src)

            mat_dict = {
                'H': Hb,
                'S': Sb,
                'F': Fb,
                'P': Pb,
                'F_H': Fb - Hb,
                'score': self_block['score'],
            }

            if method == "pbe":
                diag_data = diag_blocks[z_src]
                H_red = permute_block(H_blc - diag_data["H"], z_src, z_src)
                F_red = permute_block(F_blc - diag_data["F"], z_src, z_src)
                P_red = permute_block(P_blc - diag_data["P"], z_src, z_src)
                mat_dict['H_red'] = H_red
                mat_dict['F_red'] = F_red
                mat_dict['P_red'] = P_red
                mat_dict['F_H_red'] = F_red - H_red

            blocks.append({
                "is_self": True,
                "ctr": self_ctr,
                "source": src,
                "neighbor": src,
                "cell": self_block['cell'],
                "matrix": mat_dict,
            })
            self_ctr += 1

        # Select neighbor blocks: distance-based or score-based
        if use_dist:
            # Distances already computed during accumulation
            neighbor_blocks.sort(key=lambda b: b['_dist'])
            if len(neighbor_blocks) > topk:
                radius = neighbor_blocks[topk - 1]['_dist']
                selected = [b for b in neighbor_blocks if b['_dist'] <= radius * (1 + 1e-10)]
            else:
                selected = neighbor_blocks
        else:
            neighbor_blocks.sort(key=lambda b: b['score'], reverse=True)
            if len(neighbor_blocks) > topk:
                topk_score = neighbor_blocks[topk - 1]['score']
                selected = neighbor_blocks[:topk]
                #[b for b in neighbor_blocks if b['score'] >= topk_score * (1 - 1e-6)]
            else:
                selected = neighbor_blocks
            #max_nbr_selected = max(max_nbr_selected, len(selected))

        for block in selected:
            ngb = block['ngb']
            mats = block['mats']

            ic1, ic2, ic3 = block['cell']
            z_src = int(elem_numbers[src - 1])
            z_ngb = int(elem_numbers[ngb - 1])

            # Use precomputed distance if available, else compute now
            if '_dist' in block:
                dist = block['_dist']
            else:
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

    #if not use_dist:
        #print(f"[score selection] max neighbor blocks chosen: {max_nbr_selected} (topk={topk})")

    return blocks


def convert_kspace_blocks(data: Dict, norb_by_z: Dict[int, int], method: str = "pbe") -> list:
    """
    Convert k-space matrices to atom-pair blocks, mirroring R-space block extraction.

    For every k-point and every (src, ngb) atom pair, one block is produced:
      - src == ngb  →  self block
      - src != ngb  →  pair block
    All blocks are included (no score-based top-k cutoff).
    The 'cell' field holds the k-point coordinate (kx, ky, kz) [2π/Bohr]
    instead of a lattice translation vector.

    Args:
        data: pickle dict with 'matrices', 'kpoints_2pi_bohr', 'atomic_numbers', 'nao'
        norb_by_z: atomic-number → number of orbitals mapping
        method: 'pbe' or 'xtb' (controls which permutation is applied)

    Returns:
        List of block dicts with the same schema as convert_matrices_to_blocks.
    """
    if method == "pbe":
        permute_block = _permute_block_pbe
    elif method == "scan":
        permute_block = _permute_block_scan
    elif method == "tzvp":
        permute_block = _permute_block_tzvp
    else:
        permute_block = _permute_block

    K_matrices = data["matrices"]
    kpoints = data["kpoints_2pi_bohr"]  # (nK, 3)
    elem_numbers = data["atomic_numbers"]
    coords_bohr = data["geometry_bohr"]
    n_atoms = len(elem_numbers)

    atom_ranges = []
    offset = 0
    for z in elem_numbers:
        norb = norb_by_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    def get_block(mat, src, ngb):
        r_start, r_end = atom_ranges[src - 1]
        c_start, c_end = atom_ranges[ngb - 1]
        if issparse(mat):
            return mat[r_start:r_end, c_start:c_end].toarray()
        return mat[r_start:r_end, c_start:c_end]

    blocks = []
    self_ctr, pair_ctr = 0, 0

    for K_vec in kpoints:
        key = tuple(K_vec)
        mats = K_matrices[key]
        F_K, P_K, S_K, H_K = mats["F"], mats["P"], mats["S"], mats["H"]

        for src in range(1, n_atoms + 1):
            for ngb in range(1, n_atoms + 1):
                is_self = (src == ngb)
                z_src = int(elem_numbers[src - 1])
                z_ngb = int(elem_numbers[ngb - 1])

                Hb = permute_block(get_block(H_K, src, ngb), z_src, z_ngb)
                Fb = permute_block(get_block(F_K, src, ngb), z_src, z_ngb)
                Sb = permute_block(get_block(S_K, src, ngb), z_src, z_ngb)
                Pb = permute_block(get_block(P_K, src, ngb), z_src, z_ngb)

                entry = {
                    "is_self": is_self,
                    "ctr": self_ctr if is_self else pair_ctr,
                    "source": src,
                    "neighbor": ngb,
                    "cell": key,  # (kx, ky, kz) [2pi/Bohr]
                    "matrix": {"H": Hb, "S": Sb, "F": Fb, "P": Pb, "F_H": Fb - Hb},
                }
                if not is_self:
                    r_i = coords_bohr[src - 1]
                    r_j = coords_bohr[ngb - 1]
                    entry["dist"] = float(np.linalg.norm(r_j - r_i))
                blocks.append(entry)

                if is_self:
                    self_ctr += 1
                else:
                    pair_ctr += 1

    return blocks


def load_npz_pbc(npz_path: Path, npz, norb_by_z: Dict[int, int] = None, method: str = "xtb", topk: int = 32, use_dist: bool = False) -> Dict[str, object]:
    """Load NPZ or pickle file for periodic (PBC) materials.

    For R-space pickles (rspace=True): converts T-matrices to atom-pair blocks (top-k).
    For K-space pickles (rspace=False): stacks full matrices over k-points, no block extraction.
    Mode is auto-detected from the 'rspace' key in the pickle (defaults to True for
    backwards compatibility with pickles that predate the rspace flag).
    """
    if npz:
        with np.load(npz_path, allow_pickle=True) as z:
            return {
                "rspace": True,
                "atomic_numbers": z["atomic_numbers"],
                "geometry_bohr": z["geometry_bohr"],
                "net_spin": int(z["net_spin"]) if "net_spin" in z.files else 0,
                "charge": int(z["charge"]),
                "cell_bohr": z["cell_bohr"],
                "pbc": z["pbc"],
                "energy_xtb_Ha": float(z["energy_xtb_Ha"]) if "energy_xtb_Ha" in z.files else None,
                "blocks": z["blocks"],
                "bandgap_pbe": float(z["bandgap_pbe"]) if "bandgap_pbe" in z.files else None,
                "bandgap_hse": float(z["bandgap_hse"]) if "bandgap_hse" in z.files else None,
                "bandgap_cp2k": float(z["bandgap_cp2k"]) if "bandgap_cp2k" in z.files else None,
                "gap_type_pbe": str(z["gap_type_pbe"]) if "gap_type_pbe" in z.files else None,
                "gap_type_hse": str(z["gap_type_hse"]) if "gap_type_hse" in z.files else None,
                "m_charges": None,
                "h_charges": None,
            }

    # --- Pickle path ---
    with open(npz_path, "rb") as f:
        data = pickle.load(f)

    rspace = True#data["rspace"]
    #print(f"[ATTENTION] The model is hardfixed to rspace!!!")

    element_numbers = data["atomic_numbers"]
    coords_bohr = data["geometry_bohr"]
    net_spin = int(data["net_spin"])
    charge = int(data["charge"])
    cell_bohr = data["cell_bohr"]
    pbc = data["pbc"]
    energy = float(data["energy_cp2k_Ha"]) if "energy_cp2k_Ha" in data else None
    bandgap_pbe = float(data["bandgap_pbe"]) if "bandgap_pbe" in data else None
    bandgap_hse = float(data["bandgap_hse"]) if "bandgap_hse" in data else None
    bandgap_cp2k = float(data["bandgap_cp2k"]) if "bandgap_cp2k" in data else None
    #gap_type_pbe = str(data["gap_type_pbe"]) if "gap_type_pbe" in data else None
    #gap_type_hse = str(data["gap_type_hse"]) if "gap_type_hse" in data else None

    if rspace:
        blocks = convert_matrices_to_blocks(data, norb_by_z, method=method, topk=topk, use_dist=use_dist)
    else:
        # no top-k cutoff
        blocks = convert_kspace_blocks(data, norb_by_z, method=method)

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
        "rspace": rspace,
        "atomic_numbers": element_numbers,
        "geometry_bohr": coords_bohr,
        "charge": charge,
        "net_spin": net_spin,
        "energy_cp2k_Ha": energy,
        "cell_bohr": cell_bohr,
        "pbc": pbc,
        "blocks": blocks,
        "bandgap_pbe": bandgap_pbe,
        "bandgap_hse": bandgap_hse,
        "bandgap_cp2k": bandgap_cp2k,
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
                energy_xtb_Ha
                bandgap_pbe_eV
                bandgap_hse_eV
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
    if payload["m_charges"] is not None:
        geom0.create_dataset("m_charges", data=payload["m_charges"], dtype="f8")
    if payload["h_charges"] is not None:
        geom0.create_dataset("h_charges", data=payload["h_charges"], dtype="f8")
    geom0.create_dataset("net_spin", data=payload["net_spin"], dtype="f8")
    geom0.create_dataset("cell_bohr", data=payload["cell_bohr"], dtype="f8")
    geom0.create_dataset("pbc", data=payload["pbc"], dtype="i8")

    if payload["energy_cp2k_Ha"] is not None:
        geom0.create_dataset("energy_cp2k_Ha", data=payload["energy_cp2k_Ha"], dtype="f8")

    if payload["bandgap_pbe"] is not None:
        geom0.create_dataset("bandgap_pbe_eV", data=payload["bandgap_pbe"], dtype="f8")
    if payload["bandgap_hse"] is not None:
        geom0.create_dataset("bandgap_hse_eV", data=payload["bandgap_hse"], dtype="f8")
    if payload["bandgap_cp2k"] is not None:
        geom0.create_dataset("bandgap_cp2k_eV", data=payload["bandgap_cp2k"], dtype="f8")
    #if payload["gap_type_pbe"] is not None:
    #    geom0.attrs["gap_type_pbe"] = payload["gap_type_pbe"]
    #if payload["gap_type_hse"] is not None:
    #    geom0.attrs["gap_type_hse"] = payload["gap_type_hse"]

    # Process blocks
    blocks = payload["blocks"]

    el_numbers = payload["atomic_numbers"]

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
            dist = block["dist"]
            score = mat["score"]
            pair_rows.append([ctr, src_idx, nbr_idx, int(ic1), int(ic2), int(ic3), score, dist])

        p = geom0.create_group(name)
        p0 = p.create_group("2body")

        p0.create_dataset("H", data=mat["H"], dtype="f8")
        p0.create_dataset("S", data=mat["S"], dtype="f8")
        p0.create_dataset("F", data=mat["F"], dtype="f8")
        p0.create_dataset("P", data=mat["P"], dtype="f8")
        p0.create_dataset("F_H", data=mat["F_H"], dtype="f8")
        if "H_red" in mat:
            p0.create_dataset("H_red", data=mat["H_red"], dtype="f8")
            p0.create_dataset("F_red", data=mat["F_red"], dtype="f8")
            p0.create_dataset("P_red", data=mat["P_red"], dtype="f8")
            p0.create_dataset("F_H_red", data=mat["F_H_red"], dtype="f8")
        #REMEMBER TO CHECK ONE-BODY AND SYMMETRICITY

    # Write index arrays
    if self_rows:
        geom0.create_dataset("self_idx", data=np.array(self_rows, dtype="int64"), dtype="i8")
    if pair_rows:
        pair_arr = np.array([tuple(row) for row in pair_rows], dtype="f8")
        geom0.create_dataset("pair_idx", data=pair_arr)


def write_kspace_material(
    h5_group: h5py.Group,
    material_id: str,
    payload: Dict[str, object],
) -> None:
    """
    Write a single k-indexed material to an HDF5 group. 
    #NOTE: This is a naive implementation

    Mirrors write_pbc_material exactly, except:
      - kpoints_2pi_bohr is stored instead of translation-vector metadata
      - self_idx / pair_idx rows are [ctr, src, ngb, kx, ky, kz] (float, k-coords in 2π/Bohr)
      - Complex matrices
      - All atom-pair blocks at all k-points are present (no top-k selection)

    Structure:
        {material_id}/
            geom0/
                atomic_numbers, geometry_bohr, charge, net_spin, cell_bohr, pbc
                energy_cp2k_Ha  (optional)
                kpoints_2pi_bohr  (nK, 3)
                bandgap_*_eV  (optional)
                self_idx   (n_self, 6) float64: [ctr, src, ngb, kx, ky, kz]
                pair_idx   (n_pair, 7) float64: [ctr, src, ngb, kx, ky, kz, dist]
                self_{i}/2body/H, S, F, P, F_H   complex64
                pair_{i}/2body/H, S, F, P, F_H   complex64
    """
    grp_mat = h5_group.create_group(material_id)
    geom0 = grp_mat.create_group("geom0")

    geom0.create_dataset("atomic_numbers", data=payload["atomic_numbers"], dtype="i8")
    geom0.create_dataset("geometry_bohr", data=payload["geometry_bohr"], dtype="f8")
    geom0.create_dataset("charge", data=payload["charge"], dtype="f8")
    geom0.create_dataset("net_spin", data=payload["net_spin"], dtype="f8")
    geom0.create_dataset("cell_bohr", data=payload["cell_bohr"], dtype="f8")
    geom0.create_dataset("pbc", data=payload["pbc"], dtype="i8")
    geom0.create_dataset("kpoints_2pi_bohr", data=payload["kpoints_2pi_bohr"], dtype="f8")

    if payload["energy_cp2k_Ha"] is not None:
        geom0.create_dataset("energy_cp2k_Ha", data=payload["energy_cp2k_Ha"], dtype="f8")
    if payload["bandgap_pbe"] is not None:
        geom0.create_dataset("bandgap_pbe_eV", data=payload["bandgap_pbe"], dtype="f8")
    if payload["bandgap_hse"] is not None:
        geom0.create_dataset("bandgap_hse_eV", data=payload["bandgap_hse"], dtype="f8")
    if payload["bandgap_cp2k"] is not None:
        geom0.create_dataset("bandgap_cp2k_eV", data=payload["bandgap_cp2k"], dtype="f8")

    blocks = payload["blocks"]
    self_rows = []
    pair_rows = []

    for block in blocks:
        is_self = block["is_self"]
        ctr = block["ctr"]
        src_idx = int(block["source"]) - 1
        nbr_idx = int(block["neighbor"]) - 1
        kx, ky, kz = block["cell"]
        mat = block["matrix"]

        if is_self:
            name = f"self_{int(ctr)}"
            self_rows.append([ctr, src_idx, nbr_idx, kx, ky, kz])
        else:
            name = f"pair_{int(ctr)}"
            dist = block["dist"]
            pair_rows.append([ctr, src_idx, nbr_idx, kx, ky, kz, dist])

        p = geom0.create_group(name)
        p0 = p.create_group("2body")
        p0.create_dataset("H", data=mat["H"])
        p0.create_dataset("S", data=mat["S"])
        p0.create_dataset("F", data=mat["F"])
        p0.create_dataset("P", data=mat["P"])
        p0.create_dataset("F_H", data=mat["F_H"])

    if self_rows:
        geom0.create_dataset("self_idx", data=np.array(self_rows, dtype="f8"))
    if pair_rows:
        geom0.create_dataset("pair_idx", data=np.array(pair_rows, dtype="f8"))


def _load_material_worker(args: Tuple) -> Tuple[str, Optional[Dict]]:
    """
    Worker function for parallel loading of materials.

    Args:
        args: Tuple of (mat_id, npz_path, npz, norb_by_z, method, topk)

    Returns:
        Tuple of (mat_id, payload) or (mat_id, None) if loading fails
    """
    mat_id, npz_path, npz, norb_by_z, method, topk, use_dist = args
    payload = load_npz_pbc(npz_path, npz, norb_by_z=norb_by_z, method=method, topk=topk, use_dist=use_dist)
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
    use_dist: bool = False,
) -> int:
    """
    Write a train/val/test split for PBC materials.
    """
    split_grp = h5_file.create_group(split_name)
    count = 0

    # Filter to valid material IDs
    valid_ids = [mid for mid in material_ids if mid in id_to_npz]

    def _write(mat_id, payload):
        if payload["rspace"]:
            write_pbc_material(split_grp, mat_id, payload)
        else:
            write_kspace_material(split_grp, mat_id, payload)

    if n_workers > 1:
        # Parallel loading with sequential writes
        args_list = [
            (mat_id, id_to_npz[mat_id], npz, norb_by_z, method, topk, use_dist)
            for mat_id in valid_ids
        ]

        with Pool(n_workers) as pool:
            results = pool.imap(_load_material_worker, args_list)
            for mat_id, payload in tqdm(results, total=len(args_list), desc=f"[{split_name}]"):
                if payload is not None:
                    _write(mat_id, payload)
                    count += 1
    else:
        # Sequential loading and writing
        for mat_id in tqdm(valid_ids, desc=f"[{split_name}]"):
            npz_path = id_to_npz[mat_id]
            payload = load_npz_pbc(npz_path, npz, norb_by_z=norb_by_z, method=method, topk=topk, use_dist=use_dist)
            _write(mat_id, payload)
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


### METHODS for CP2K molecules (non-periodic, method=xyz) ###

def load_pickle_xyz_payload(pkl_path: Path) -> Dict[str, object]:
    """Load molecule data from a pickle file produced by postproc_matrices(method='xyz')."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    atomic_numbers = data["atomic_numbers"]
    perm_map = get_perm_map(atomic_numbers)
    F = apply_perm_map(data["F"], perm_map)
    P = apply_perm_map(data["P"], perm_map)
    S = apply_perm_map(data["S"], perm_map)
    H = apply_perm_map(data["H"], perm_map)
    return {
        "atomic_numbers":atomic_numbers,
        "geometry_bohr": data["geometry_bohr"],
        "charge": int(data["charge"]),
        # energy_cp2k_Ha is the xTB energy computed by CP2K for the xyz template
        "energy_xtb_Ha": float(data["energy_cp2k_Ha"]) if "energy_cp2k_Ha" in data else None,
        "energy_dft_Ha": float(data["energy_dft_Ha"]) if "energy_dft_Ha" in data else None,
        "F": F,
        "P": P,
        "S": S,
        "H": H,
    }


def write_one_xyz_molecule(
    split_grp: h5py.Group,
    mol_name: str,
    payload: Dict[str, object],
    net_spin: int = 0,
) -> None:
    """Write a single non-periodic CP2K molecule to an HDF5 split group."""
    grp = split_grp.create_group(mol_name)
    geo = grp.create_group("0")

    geo.create_dataset("atomic_numbers", data=payload["atomic_numbers"])
    geo.create_dataset("geometry_bohr", data=payload["geometry_bohr"], dtype="f8")
    geo.create_dataset("charge", data=int(payload["charge"]))
    geo.create_dataset("net_spin", data=net_spin)

    if payload.get("energy_xtb_Ha") is not None:
        geo.create_dataset("energy_xtb_Ha", data=float(payload["energy_xtb_Ha"]), dtype="f8")
    if payload.get("energy_dft_Ha") is not None:
        geo.create_dataset("energy_dft_Ha", data=float(payload["energy_dft_Ha"]), dtype="f8")

    two_body = geo.create_group("2body")
    two_body.create_dataset("F", data=payload["F"], dtype="f8")
    two_body.create_dataset("P", data=payload["P"], dtype="f8")
    two_body.create_dataset("S", data=payload["S"], dtype="f8")
    two_body.create_dataset("H", data=payload["H"], dtype="f8")
    two_body.create_dataset("F_H", data=payload["F"] - payload["H"], dtype="f8")
    if "H_red" in payload:
        two_body.create_dataset("H_red", data=payload["H_red"], dtype="f8")
        two_body.create_dataset("F_red", data=payload["F_red"], dtype="f8")
        two_body.create_dataset("P_red", data=payload["P_red"], dtype="f8")
        two_body.create_dataset("F_H_red", data=payload["F_red"] - payload["H_red"], dtype="f8")


def write_xyz_split(
    h5_file: h5py.File,
    split_name: str,
    material_ids: Iterable[str],
    id_to_pkl: Dict[str, Path],
    net_spin: int = 0,
) -> int:
    """Write a train/val/test split for non-periodic (xyz) molecules from CP2K pickles."""
    split_grp = h5_file.create_group(split_name)
    count = 0
    valid_ids = [mid for mid in material_ids if mid in id_to_pkl]

    for mol_id in tqdm(valid_ids, desc=f"[{split_name}]"):
        pkl_path = id_to_pkl[mol_id]
        payload = load_pickle_xyz_payload(pkl_path)
        write_one_xyz_molecule(split_grp, mol_id, payload, net_spin=net_spin)
        count += 1

    return count


### METHODS for pbemol: CP2K non-periodic molecules with PBE basis ###

def load_pickle_pbemol_payload(pkl_path: Path) -> Dict[str, object]:
    """Load pbemol molecule data from a pickle and apply PBE permutations block-wise."""
    from .postproc import norb_by_z as _norb_by_z

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    elem_numbers = data["atomic_numbers"]
    norb_z = _norb_by_z("pbe")

    atom_ranges = []
    offset = 0
    for z in elem_numbers:
        norb = norb_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    def permute_full(mat):
        dense = np.array(mat, dtype=np.float64)
        out = np.empty_like(dense)
        for i, (rs, re) in enumerate(atom_ranges):
            for j, (cs, ce) in enumerate(atom_ranges):
                out[rs:re, cs:ce] = _permute_block_pbe(dense[rs:re, cs:ce], elem_numbers[i], elem_numbers[j])
        return out

    raw_F = np.array(data["F"], dtype=np.float64)
    raw_P = np.array(data["P"], dtype=np.float64)
    raw_S = np.array(data["S"], dtype=np.float64)
    raw_H = np.array(data["H"], dtype=np.float64)

    # Compute reduced versions by subtracting atomic diagonal blocks (in raw space)
    raw_H_red = raw_H.copy()
    raw_F_red = raw_F.copy()
    raw_P_red = raw_P.copy()
    for i, (rs, re) in enumerate(atom_ranges):
        z = int(elem_numbers[i])
        raw_H_red[rs:re, rs:re] -= diag_blocks[z]["H"]
        raw_F_red[rs:re, rs:re] -= diag_blocks[z]["F"]
        raw_P_red[rs:re, rs:re] -= diag_blocks[z]["P"]

    F = permute_full(raw_F)
    P = permute_full(raw_P)
    S = permute_full(raw_S)
    H = permute_full(raw_H)
    F_red = permute_full(raw_F_red)
    P_red = permute_full(raw_P_red)
    H_red = permute_full(raw_H_red)

    return {
        "atomic_numbers": elem_numbers,
        "geometry_bohr": data["geometry_bohr"],
        "charge": int(data["charge"]),
        "energy_xtb_Ha": float(data["energy_cp2k_Ha"]) if "energy_cp2k_Ha" in data else None,
        "energy_dft_Ha": float(data["energy_dft_Ha"]) if "energy_dft_Ha" in data else None,
        "F": F,
        "P": P,
        "S": S,
        "H": H,
        "F_red": F_red,
        "P_red": P_red,
        "H_red": H_red,
    }


def write_pbemol_split(
    h5_file: h5py.File,
    split_name: str,
    material_ids: Iterable[str],
    id_to_pkl: Dict[str, Path],
    net_spin: int = 0,
) -> int:
    """Write a train/val/test split for pbemol molecules from CP2K pickles."""
    split_grp = h5_file.create_group(split_name)
    count = 0
    errors = 0
    valid_ids = [mid for mid in material_ids if mid in id_to_pkl]

    for mol_id in tqdm(valid_ids, desc=f"[{split_name}]"):
        pkl_path = id_to_pkl[mol_id]
        try:
            payload = load_pickle_pbemol_payload(pkl_path)
        except Exception as e:
            print(f"[WARN] Skipping {mol_id}: {e}")
            errors += 1
            continue
        write_one_xyz_molecule(split_grp, mol_id, payload, net_spin=net_spin)
        count += 1

    if errors:
        print(f"[{split_name}] {errors} entries skipped due to errors.")
    return count


### METHODS for xtb_mol: periodic xtb pickle -> molecular format via T=(0,0,0) ###

def load_pickle_xtb_mol_payload(pkl_path: Path) -> Dict[str, object]:
    """
    Extract the T=(0,0,0) block from a periodic xtb R-space pickle and package it
    as a molecular (xyz-format) payload for testing the molecular model on crystal features.
    Applies _permute_block per atom-pair block — identical to the periodic xtb pipeline.
    """
    from .postproc import norb_by_z as _norb_by_z

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    t0_key = (0, 0, 0)
    mats = data["matrices"].get(t0_key)
    if mats is None:
        raise KeyError(f"T=(0,0,0) not found in matrices of {pkl_path}")

    elem_numbers = data["atomic_numbers"]
    norb_z = _norb_by_z("xtb")

    # Build orbital ranges per atom
    atom_ranges = []
    offset = 0
    for z in elem_numbers:
        norb = norb_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    def permute_full(mat):
        """Densify and apply _permute_block to every atom-pair block."""
        dense = mat.toarray().astype(np.float64) if issparse(mat) else np.array(mat, dtype=np.float64)
        out = np.empty_like(dense)
        for i, (rs, re) in enumerate(atom_ranges):
            for j, (cs, ce) in enumerate(atom_ranges):
                out[rs:re, cs:ce] = _permute_block(dense[rs:re, cs:ce], elem_numbers[i], elem_numbers[j])
        return out

    F = permute_full(mats["F"])
    P = permute_full(mats["P"])
    S = permute_full(mats["S"])
    H = permute_full(mats["H"])

    return {
        "atomic_numbers": elem_numbers,
        "geometry_bohr": data["geometry_bohr"],
        "charge": int(data["charge"]),
        "net_spin": int(data.get("net_spin", 0)),
        "energy_cp2k_Ha": float(data["energy_cp2k_Ha"]) if "energy_cp2k_Ha" in data else None,
        "bandgap_pbe": float(data["bandgap_pbe"]) if data.get("bandgap_pbe") is not None else None,
        "bandgap_hse": float(data["bandgap_hse"]) if data.get("bandgap_hse") is not None else None,
        "bandgap_cp2k": float(data["bandgap_cp2k"]) if data.get("bandgap_cp2k") is not None else None,
        "F": F,
        "P": P,
        "S": S,
        "H": H,
    }


def write_one_xtb_mol(
    split_grp: h5py.Group,
    mol_name: str,
    payload: Dict[str, object],
) -> None:
    """
    Write a single xtb_mol entry (crystal packed as a molecule) to an HDF5 split group.
    Format mirrors write_one_xyz_molecule but also stores bandgap labels when available.
    """
    grp = split_grp.create_group(mol_name)
    geo = grp.create_group("0")

    geo.create_dataset("atomic_numbers", data=payload["atomic_numbers"])
    geo.create_dataset("geometry_bohr", data=payload["geometry_bohr"], dtype="f8")
    geo.create_dataset("charge", data=int(payload["charge"]))
    geo.create_dataset("net_spin", data=int(payload["net_spin"]))

    if payload.get("energy_cp2k_Ha") is not None:
        geo.create_dataset("energy_cp2k_Ha", data=float(payload["energy_cp2k_Ha"]), dtype="f8")
    if payload.get("bandgap_pbe") is not None:
        geo.create_dataset("bandgap_pbe_eV", data=float(payload["bandgap_pbe"]), dtype="f8")
    if payload.get("bandgap_hse") is not None:
        geo.create_dataset("bandgap_hse_eV", data=float(payload["bandgap_hse"]), dtype="f8")
    if payload.get("bandgap_cp2k") is not None:
        geo.create_dataset("bandgap_cp2k_eV", data=float(payload["bandgap_cp2k"]), dtype="f8")

    two_body = geo.create_group("2body")
    two_body.create_dataset("F", data=payload["F"], dtype="f8")
    two_body.create_dataset("P", data=payload["P"], dtype="f8")
    two_body.create_dataset("S", data=payload["S"], dtype="f8")
    two_body.create_dataset("H", data=payload["H"], dtype="f8")
    two_body.create_dataset("F_H", data=payload["F"] - payload["H"], dtype="f8")


def write_xtb_mol_split(
    h5_file: h5py.File,
    split_name: str,
    material_ids: Iterable[str],
    id_to_pkl: Dict[str, Path],
) -> int:
    """Write a train/val/test split for xtb_mol entries (periodic xtb -> molecular format)."""
    split_grp = h5_file.create_group(split_name)
    count = 0
    errors = 0
    valid_ids = [mid for mid in material_ids if mid in id_to_pkl]

    for mat_id in tqdm(valid_ids, desc=f"[{split_name}]"):
        pkl_path = id_to_pkl[mat_id]
        try:
            payload = load_pickle_xtb_mol_payload(pkl_path)
        except Exception as e:
            print(f"[WARN] Skipping {mat_id}: {e}")
            errors += 1
            continue
        write_one_xtb_mol(split_grp, mat_id, payload)
        count += 1

    if errors:
        print(f"[{split_name}] {errors} entries skipped due to errors.")
    return count


### METHODS for xtb_super: build supercell molecular representation from R-space matrices ###

def load_pickle_xtb_super_payload(pkl_path: Path, super_size=(2, 2, 2)) -> Dict[str, object]:
    """
    Build an (L1 x L2 x L3) supercell Hamiltonian from a periodic xtb R-space pickle,
    then package it as a molecular payload (same format as xtb_mol).

    The block between atom i in unit cell (a1,b1,c1) and atom j in unit cell (a2,b2,c2)
    is taken directly from the R-space matrix at T = (a2-a1, b2-b2, c2-c1). T-vectors
    not present in the pickle contribute zero (interactions decayed beyond that range).

    Efficiency: each unit-cell matrix at each needed T is permuted once (O(n_atoms_uc^2)
    block ops per T), then tiled into the supercell, avoiding an O(n_atoms_super^2) loop.
    """
    from .postproc import norb_by_z as _norb_by_z

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    matrices = data["matrices"]
    elem_numbers_uc = data["atomic_numbers"]
    coords_bohr_uc = data["geometry_bohr"]
    cell_bohr = data["cell_bohr"]

    norb_z = _norb_by_z("xtb")
    nao_uc = sum(norb_z[int(z)] for z in elem_numbers_uc)

    L1, L2, L3 = super_size
    n_cells = L1 * L2 * L3

    # Orbital ranges within one unit cell
    atom_ranges_uc = []
    offset = 0
    for z in elem_numbers_uc:
        norb = norb_z[int(z)]
        atom_ranges_uc.append((offset, offset + norb))
        offset += norb

    def permute_uc_matrix(mat):
        """Apply _permute_block to every atom-pair block of a unit-cell matrix."""
        dense = mat.toarray().astype(np.float64) if issparse(mat) else np.array(mat, dtype=np.float64)
        out = np.empty_like(dense)
        for i, (rs, re) in enumerate(atom_ranges_uc):
            for j, (cs, ce) in enumerate(atom_ranges_uc):
                out[rs:re, cs:ce] = _permute_block(dense[rs:re, cs:ce], elem_numbers_uc[i], elem_numbers_uc[j])
        return out

    # All unit cells in the supercell
    cells = [(a, b, c) for a in range(L1) for b in range(L2) for c in range(L3)]

    # Pre-permute only the T-vectors we actually need
    needed_Ts = {
        (cell_J[0] - cell_I[0], cell_J[1] - cell_I[1], cell_J[2] - cell_I[2])
        for cell_I in cells
        for cell_J in cells
    }

    permuted = {}
    # Load and permute only T-vectors with all non-negative components
    for T in needed_Ts:
        if any(t < 0 for t in T):
            continue
        mats = matrices.get(T)
        if mats is None:
            continue
        permuted[T] = {key: permute_uc_matrix(mats[key]) for key in ("F", "P", "S", "H")}

    # Derive negative T-vectors via transpose: H[-R] = H[R]^T
    for T in needed_Ts:
        if T in permuted:
            continue
        neg_T = (-T[0], -T[1], -T[2])
        pm_pos = permuted.get(neg_T)
        if pm_pos is None:
            continue
        permuted[T] = {key: pm_pos[key].T for key in ("F", "P", "S", "H")}

    # Assemble supercell matrices
    nao_super = nao_uc * n_cells
    F_s = np.zeros((nao_super, nao_super), dtype=np.float64)
    P_s = np.zeros_like(F_s)
    S_s = np.zeros_like(F_s)
    H_s = np.zeros_like(F_s)

    for idx_I, cell_I in enumerate(cells):
        row_off = idx_I * nao_uc
        for idx_J, cell_J in enumerate(cells):
            T = (cell_J[0] - cell_I[0], cell_J[1] - cell_I[1], cell_J[2] - cell_I[2])
            pm = permuted.get(T)
            if pm is None:
                continue
            col_off = idx_J * nao_uc
            F_s[row_off:row_off + nao_uc, col_off:col_off + nao_uc] = pm["F"]
            P_s[row_off:row_off + nao_uc, col_off:col_off + nao_uc] = pm["P"]
            S_s[row_off:row_off + nao_uc, col_off:col_off + nao_uc] = pm["S"]
            H_s[row_off:row_off + nao_uc, col_off:col_off + nao_uc] = pm["H"]

    # Supercell geometry: tile unit cell atoms with lattice shifts
    a1, a2, a3 = cell_bohr[0], cell_bohr[1], cell_bohr[2]
    coords_super = np.vstack([
        coords_bohr_uc + ia * a1 + ib * a2 + ic * a3
        for (ia, ib, ic) in cells
    ])
    elem_numbers_super = np.tile(elem_numbers_uc, n_cells)

    return {
        "atomic_numbers": elem_numbers_super,
        "geometry_bohr": coords_super,
        "charge": int(data["charge"]) * n_cells,
        "net_spin": int(data.get("net_spin", 0)) * n_cells,
        "energy_cp2k_Ha": float(data["energy_cp2k_Ha"]) * n_cells if "energy_cp2k_Ha" in data else None,
        "bandgap_pbe": float(data["bandgap_pbe"]) if data.get("bandgap_pbe") is not None else None,
        "bandgap_hse": float(data["bandgap_hse"]) if data.get("bandgap_hse") is not None else None,
        "bandgap_cp2k": float(data["bandgap_cp2k"]) if data.get("bandgap_cp2k") is not None else None,
        "F": F_s,
        "P": P_s,
        "S": S_s,
        "H": H_s,
    }


def write_xtb_super_split(
    h5_file: h5py.File,
    split_name: str,
    material_ids: Iterable[str],
    id_to_pkl: Dict[str, Path],
    super_size: Tuple[int, int, int] = (2, 2, 2),
) -> int:
    """Write a train/val/test split of supercell xtb entries in molecular HDF5 format."""
    split_grp = h5_file.create_group(split_name)
    count = 0
    errors = 0
    valid_ids = [mid for mid in material_ids if mid in id_to_pkl]

    for mat_id in tqdm(valid_ids, desc=f"[{split_name}]"):
        pkl_path = id_to_pkl[mat_id]
        try:
            payload = load_pickle_xtb_super_payload(pkl_path, super_size=super_size)
        except Exception as e:
            print(f"[WARN] Skipping {mat_id}: {e}")
            errors += 1
            continue
        write_one_xtb_mol(split_grp, mat_id, payload)  # same HDF5 format as xtb_mol
        count += 1

    if errors:
        print(f"[{split_name}] {errors} entries skipped due to errors.")
    return count
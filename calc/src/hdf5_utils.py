"""HDF5 dataset builder for CP2K matrices.

Periodic materials use a columnar block format. Block topology and flat matrix data
are stored as 1-D arrays with a CSR-style offset index.

Molecular materials store full (nao × nao) dense matrices directly.

HDF5 layout — periodic:
    material_id/
        atomic_numbers   int8  (n_atoms,)
        geometry_bohr    float32 (n_atoms, 3)
        cell_bohr        float32 (3, 3)
        pbc              bool (3,)
        m_charges        float32 (n_atoms,)   [optional]
        h_charges        float32 (n_atoms,)   [optional]
        block_src        int16  (n_blocks,)
        block_ngb        int16  (n_blocks,)
        block_T          int16 or float32 (n_blocks, 3)
        block_is_self    uint8  (n_blocks,)
        block_dist       float32 (n_blocks,)
        block_offsets    int32  (n_blocks+1,)
        block_shapes     int16  (n_blocks, 2)
        H, S, F, P, F_H  float32 (total_elements,)   [gzip]
        attrs: charge, net_spin, rspace, energy_cp2k_Ha, bandgap_*

HDF5 layout — molecular:
    material_id/
        atomic_numbers   int8  (n_atoms,)
        geometry_bohr    float32 (n_atoms, 3)
        H, S, F, P, F_H  float32 (nao, nao)
        attrs: charge, energy_*, bandgap_*
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
from scipy.sparse import issparse
from tqdm.auto import tqdm

from .config import METHODS
from .npz_io import read_periodic, read_molecular
from .perm_and_blocks import (
    _permute_block, _permute_block_pbe, _permute_block_scan, _permute_block_tzvp,
)

# ── Permutation dispatch ──────────────────────────────────────────────────────

def _get_permute_fn(method: str):
    perm = METHODS[method].orbital_perm if method in METHODS else "xtb"
    return {
        "dz":   _permute_block_pbe,
        "scan": _permute_block_scan,
        "tzvp": _permute_block_tzvp,
        "xtb":  _permute_block,
    }[perm]


# ── Splits ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


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
    n = len(arr)
    n_train = min(train_size, n)
    n_val   = min(val_size,   max(0, n - n_train))
    n_test  = min(test_size,  max(0, n - n_train - n_val))
    return SplitIndices(
        train=arr[:n_train],
        val=arr[n_train:n_train + n_val],
        test=arr[n_train + n_val:n_train + n_val + n_test],
    )


def get_npz_paths(run_dir: Path) -> Dict[str, Path]:
    """Return {material_id: matrices.npz path} for all successful calculations."""
    run_dir = Path(run_dir)
    result: Dict[str, Path] = {}
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        p = subdir / "matrices.npz"
        if p.exists():
            result[subdir.name] = p
    return result


# ── CP2K charge parsing ───────────────────────────────────────────────────────

def parse_cp2k_charges(out_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse Mulliken and Hirshfeld charges from a CP2K output file."""
    if not out_file.exists():
        return None, None

    mulliken, hirshfeld = [], []
    in_mulliken = in_hirshfeld = False

    with open(out_file) as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            if "Mulliken Population Analysis" in line:
                in_mulliken, mulliken = True, []
            elif "Hirshfeld Charges" in line:
                in_hirshfeld, hirshfeld = True, []
            elif in_mulliken:
                parts = clean.split()
                if parts[0].isdigit():
                    mulliken.append(float(parts[-1]))
                elif "Total charge" in line or "!" in line:
                    in_mulliken = False
            elif in_hirshfeld:
                parts = clean.split()
                if parts[0].isdigit():
                    hirshfeld.append(float(parts[-1]))
                elif "Total Charge" in line or "!" in line:
                    in_hirshfeld = False

    return (
        np.array(mulliken) if mulliken else None,
        np.array(hirshfeld) if hirshfeld else None,
    )


# ── Block extraction: periodic R-space ───────────────────────────────────────

def extract_rspace_blocks(
    data: dict,
    norb_by_z: Dict[int, int],
    method: str = "xtb",
    topk: int = 32,
    cutoff: float = 1e-42,
    use_dist: bool = False,
) -> list:
    """Convert R-space T-matrices to atom-pair blocks (top-k selection).

    Returns a list of block dicts, each with keys:
        is_self, source (1-indexed), neighbor (1-indexed), cell (T-tuple),
        dist (float, pair only), matrix (dict of permuted ndarrays).
    """
    permute = _get_permute_fn(method)

    T_matrices = data["matrices"]
    elem_numbers = data["atomic_numbers"]
    coords_bohr = data["geometry_bohr"]
    cell_bohr = data["cell_bohr"]
    n_atoms = len(elem_numbers)

    # Build atom → orbital range
    atom_ranges, offset = [], 0
    for z in elem_numbers:
        norb = norb_by_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    total_orbs = atom_ranges[-1][1]
    nao_from_mat = next(iter(next(iter(T_matrices.values())).values())).shape[0]
    assert total_orbs == nao_from_mat, (
        f"orbital count mismatch: norb_by_z sums to {total_orbs} "
        f"but matrix shape is ({nao_from_mat}, {nao_from_mat})"
    )
    starts = np.array([rs for rs, _ in atom_ranges], dtype=np.intp)
    orb_to_atom = np.empty(total_orbs, dtype=np.intp)
    for i, (rs, re) in enumerate(atom_ranges):
        orb_to_atom[rs:re] = i

    def _get_block(mat, src, ngb):
        rs, re = atom_ranges[src - 1]
        cs, ce = atom_ranges[ngb - 1]
        b = mat[rs:re, cs:ce]
        return b.toarray() if hasattr(b, "toarray") else b

    def _block_norms(mat) -> np.ndarray:
        norms = np.zeros((n_atoms, n_atoms))
        if issparse(mat):
            coo = mat.tocoo()
            if coo.nnz:
                np.add.at(norms, (orb_to_atom[coo.row], orb_to_atom[coo.col]), coo.data ** 2)
        else:
            sq = np.asarray(mat) ** 2
            row_sums = np.add.reduceat(sq, starts, axis=0)
            norms = np.add.reduceat(row_sums, starts, axis=1)
        return norms

    a1, a2, a3 = cell_bohr[0], cell_bohr[1], cell_bohr[2]
    blocks_by_source: dict = defaultdict(list)
    # Detect UKS from first T-matrix entry
    uks = "F_a" in next(iter(T_matrices.values()))

    if use_dist:
        buffer_size = topk * 5
        for T_vec, mats in T_matrices.items():
            ic1, ic2, ic3 = T_vec
            shift = ic1 * a1 + ic2 * a2 + ic3 * a3
            diffs = coords_bohr[np.newaxis] - coords_bohr[:, np.newaxis] + shift
            dists_T = np.linalg.norm(diffs, axis=2)
            for src0 in range(n_atoms):
                for ngb0 in range(n_atoms):
                    blocks_by_source[src0 + 1].append({
                        "ngb": ngb0 + 1, "cell": T_vec,
                        "_dist": float(dists_T[src0, ngb0]), "mats": mats,
                    })
            for src_key in range(1, n_atoms + 1):
                lst = blocks_by_source[src_key]
                if len(lst) > buffer_size:
                    lst.sort(key=lambda b: b["_dist"])
                    blocks_by_source[src_key] = lst[:buffer_size]
    else:
        for ctr, (T_vec, mats) in enumerate(T_matrices.items()):
            if ctr >= 150:
                break
            P_score = (mats["P_a"] + mats["P_b"]) if uks else mats["P"]
            F_score = mats["F_a"] if uks else mats["F"]
            scores = _block_norms(P_score) * _block_norms(mats["S"]) * _block_norms(F_score)
            for src0, ngb0 in zip(*np.where(scores >= cutoff)):
                blocks_by_source[src0 + 1].append({
                    "ngb": int(ngb0) + 1, "cell": T_vec,
                    "score": float(scores[src0, ngb0]), "mats": mats,
                })

    blocks = []
    self_ctr = pair_ctr = 0

    for src in range(1, n_atoms + 1):
        z_src = int(elem_numbers[src - 1])
        atom_blocks = blocks_by_source.get(src, [])

        # Separate self-block from pair blocks
        self_block = next((b for b in atom_blocks if b["ngb"] == src and b["cell"] == (0, 0, 0)), None)
        neighbors = [b for b in atom_blocks if not (b["ngb"] == src and b["cell"] == (0, 0, 0))]

        if self_block is not None:
            mats = self_block["mats"]
            H_blc = _get_block(mats["H"], src, src)
            mat_dict = {
                "H": permute(H_blc, z_src, z_src),
                "S": permute(_get_block(mats["S"], src, src), z_src, z_src),
            }
            if uks:
                Fa_blc = _get_block(mats["F_a"], src, src)
                Fb_blc = _get_block(mats["F_b"], src, src)
                mat_dict["F_a"] = permute(Fa_blc, z_src, z_src)
                mat_dict["F_b"] = permute(Fb_blc, z_src, z_src)
                mat_dict["P_a"] = permute(_get_block(mats["P_a"], src, src), z_src, z_src)
                mat_dict["P_b"] = permute(_get_block(mats["P_b"], src, src), z_src, z_src)
                mat_dict["F_H_a"] = mat_dict["F_a"] - mat_dict["H"]
                mat_dict["F_H_b"] = mat_dict["F_b"] - mat_dict["H"]
            else:
                F_blc = _get_block(mats["F"], src, src)
                mat_dict["F"] = permute(F_blc, z_src, z_src)
                mat_dict["P"] = permute(_get_block(mats["P"], src, src), z_src, z_src)
                mat_dict["F_H"] = mat_dict["F"] - mat_dict["H"]
            blocks.append({
                "is_self": True, "ctr": self_ctr,
                "source": src, "neighbor": src,
                "cell": self_block["cell"], "matrix": mat_dict,
            })
            self_ctr += 1

        # Select neighbor blocks
        if use_dist:
            neighbors.sort(key=lambda b: b["_dist"])
            selected = neighbors[:topk] if len(neighbors) > topk else neighbors
        else:
            neighbors.sort(key=lambda b: b["score"], reverse=True)
            if len(neighbors) > topk:
                selected = neighbors[:topk]
            else:
                selected = neighbors
                if len(selected) < topk:
                    print(f"topk {topk} not reached for atom {src} ({len(selected)} neighbors)")

        for block in selected:
            ngb = block["ngb"]
            mats = block["mats"]
            z_ngb = int(elem_numbers[ngb - 1])
            ic1, ic2, ic3 = block["cell"]

            if "_dist" in block:
                dist = block["_dist"]
            else:
                shift = ic1 * a1 + ic2 * a2 + ic3 * a3
                dist = float(np.linalg.norm(coords_bohr[ngb - 1] - coords_bohr[src - 1] + shift))

            Hb = permute(_get_block(mats["H"], src, ngb), z_src, z_ngb)
            mat_dict = {
                "H": Hb,
                "S": permute(_get_block(mats["S"], src, ngb), z_src, z_ngb),
            }
            if uks:
                Fa_b = permute(_get_block(mats["F_a"], src, ngb), z_src, z_ngb)
                Fb_b = permute(_get_block(mats["F_b"], src, ngb), z_src, z_ngb)
                mat_dict["F_a"]   = Fa_b
                mat_dict["F_b"]   = Fb_b
                mat_dict["P_a"]   = permute(_get_block(mats["P_a"], src, ngb), z_src, z_ngb)
                mat_dict["P_b"]   = permute(_get_block(mats["P_b"], src, ngb), z_src, z_ngb)
                mat_dict["F_H_a"] = Fa_b - Hb
                mat_dict["F_H_b"] = Fb_b - Hb
            else:
                Fb = permute(_get_block(mats["F"], src, ngb), z_src, z_ngb)
                mat_dict["F"]   = Fb
                mat_dict["P"]   = permute(_get_block(mats["P"], src, ngb), z_src, z_ngb)
                mat_dict["F_H"] = Fb - Hb
            blocks.append({
                "is_self": False, "ctr": pair_ctr,
                "source": src, "neighbor": ngb,
                "cell": block["cell"], "dist": dist,
                "matrix": mat_dict,
            })
            pair_ctr += 1

    return blocks


# ── Block extraction: K-space ─────────────────────────────────────────────────

def extract_kspace_blocks(
    data: dict,
    norb_by_z: Dict[int, int],
    method: str = "xtb",
) -> list:
    """Extract all (src, ngb) atom-pair blocks at every k-point. No top-k selection."""
    permute = _get_permute_fn(method)
    K_matrices = data["matrices"]
    kpoints = data["kpoints_2pi_bohr"]
    elem_numbers = data["atomic_numbers"]
    coords_bohr = data["geometry_bohr"]
    n_atoms = len(elem_numbers)

    atom_ranges, offset = [], 0
    for z in elem_numbers:
        norb = norb_by_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    nao_from_z = atom_ranges[-1][1]
    nao_from_mat = next(iter(next(iter(K_matrices.values())).values())).shape[0]
    assert nao_from_z == nao_from_mat, (
        f"orbital count mismatch: norb_by_z sums to {nao_from_z} "
        f"but matrix shape is ({nao_from_mat}, {nao_from_mat})"
    )

    def _get_block(mat, src, ngb):
        rs, re = atom_ranges[src - 1]
        cs, ce = atom_ranges[ngb - 1]
        b = mat[rs:re, cs:ce]
        return b.toarray() if hasattr(b, "toarray") else b

    uks = "F_a" in next(iter(K_matrices.values()))

    blocks = []
    self_ctr = pair_ctr = 0
    for K_vec in kpoints:
        key = tuple(K_vec)
        mats = K_matrices[key]
        for src in range(1, n_atoms + 1):
            for ngb in range(1, n_atoms + 1):
                is_self = src == ngb
                z_src = int(elem_numbers[src - 1])
                z_ngb = int(elem_numbers[ngb - 1])
                Hb = permute(_get_block(mats["H"], src, ngb), z_src, z_ngb)
                mat_dict = {
                    "H": Hb,
                    "S": permute(_get_block(mats["S"], src, ngb), z_src, z_ngb),
                }
                if uks:
                    Fa_b = permute(_get_block(mats["F_a"], src, ngb), z_src, z_ngb)
                    Fb_b = permute(_get_block(mats["F_b"], src, ngb), z_src, z_ngb)
                    mat_dict["F_a"]   = Fa_b
                    mat_dict["F_b"]   = Fb_b
                    mat_dict["P_a"]   = permute(_get_block(mats["P_a"], src, ngb), z_src, z_ngb)
                    mat_dict["P_b"]   = permute(_get_block(mats["P_b"], src, ngb), z_src, z_ngb)
                    mat_dict["F_H_a"] = Fa_b - Hb
                    mat_dict["F_H_b"] = Fb_b - Hb
                else:
                    Fb = permute(_get_block(mats["F"], src, ngb), z_src, z_ngb)
                    mat_dict["F"]   = Fb
                    mat_dict["P"]   = permute(_get_block(mats["P"], src, ngb), z_src, z_ngb)
                    mat_dict["F_H"] = Fb - Hb
                entry = {
                    "is_self": is_self,
                    "ctr": self_ctr if is_self else pair_ctr,
                    "source": src, "neighbor": ngb, "cell": key,
                    "matrix": mat_dict,
                }
                if not is_self:
                    entry["dist"] = float(np.linalg.norm(coords_bohr[ngb - 1] - coords_bohr[src - 1]))
                blocks.append(entry)
                if is_self:
                    self_ctr += 1
                else:
                    pair_ctr += 1
    return blocks


# ── Columnar packing ──────────────────────────────────────────────────────────

def _blocks_to_columnar(blocks: list, rspace: bool = True) -> dict:
    """Pack a list of block dicts into flat columnar arrays for HDF5 storage."""
    n = len(blocks)
    # Derive keys from the first block — handles both RKS and UKS transparently
    mat_keys = tuple(blocks[0]["matrix"].keys())

    src_arr  = np.empty(n, dtype=np.int16)
    ngb_arr  = np.empty(n, dtype=np.int16)
    T_parts  = []
    is_self  = np.empty(n, dtype=np.uint8)
    dist_arr = np.zeros(n, dtype=np.float32)
    shapes   = np.empty((n, 2), dtype=np.int16)
    offsets  = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0

    flat: dict = {k: [] for k in mat_keys}

    for i, block in enumerate(blocks):
        src_arr[i] = block["source"] - 1
        ngb_arr[i] = block["neighbor"] - 1
        T_parts.append(block["cell"])
        is_self[i] = block["is_self"]
        if not block["is_self"]:
            dist_arr[i] = block.get("dist", 0.0)

        mat = block["matrix"]
        h, w = mat["H"].shape
        shapes[i] = (h, w)
        offsets[i + 1] = offsets[i] + h * w

        for k in mat_keys:
            flat[k].append(mat[k].ravel().astype(np.float32) if k in mat
                           else np.zeros(h * w, dtype=np.float32))

    # T / K-vector array — integer for rspace, float for kspace
    if rspace:
        T_arr = np.array(T_parts, dtype=np.int16)
    else:
        T_arr = np.array(T_parts, dtype=np.float32)

    result = {
        "block_src":     src_arr,
        "block_ngb":     ngb_arr,
        "block_T":       T_arr,
        "block_is_self": is_self,
        "block_dist":    dist_arr,
        "block_offsets": offsets,
        "block_shapes":  shapes,
    }
    for k in mat_keys:
        result[k] = np.concatenate(flat[k]) if flat[k] else np.empty(0, dtype=np.float32)
    return result


# ── HDF5 writers ──────────────────────────────────────────────────────────────

_GZIP_LEVEL = 4
_CHUNK = 65536


def _write_attrs(grp: h5py.Group, payload: dict, keys: tuple) -> None:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            grp.attrs[k] = float(v)


def write_periodic_material(
    h5_grp: h5py.Group,
    mid: str,
    payload: dict,
    col: dict,
) -> None:
    """Write one periodic material in columnar block format."""
    g = h5_grp.create_group(mid)

    # Geometry
    g.create_dataset("atomic_numbers", data=payload["atomic_numbers"].astype(np.int8))
    g.create_dataset("geometry_bohr",  data=payload["geometry_bohr"].astype(np.float32))
    g.create_dataset("cell_bohr",      data=payload["cell_bohr"].astype(np.float32))
    g.create_dataset("pbc",            data=np.array(payload["pbc"], dtype=np.bool_))
    if payload.get("m_charges") is not None:
        g.create_dataset("m_charges", data=payload["m_charges"].astype(np.float32))
    if payload.get("h_charges") is not None:
        g.create_dataset("h_charges", data=payload["h_charges"].astype(np.float32))

    # Scalar metadata as attributes
    g.attrs["charge"]   = int(payload.get("charge", 0))
    g.attrs["net_spin"] = int(payload.get("net_spin", 0))
    g.attrs["rspace"]   = bool(payload.get("rspace", True))
    _write_attrs(g, payload, (
        "energy_cp2k_Ha", "bandgap_pbesol", "bandgap_hse", "bandgap_cp2k",
    ))

    # Block topology (small, no compression needed)
    for k in ("block_src", "block_ngb", "block_T", "block_is_self",
              "block_dist", "block_offsets", "block_shapes"):
        g.create_dataset(k, data=col[k])

    # Flat matrix data (large, compress)
    n_total = int(col["block_offsets"][-1])
    chunk = (min(_CHUNK, max(1, n_total)),)
    mat_keys = [k for k in col if k not in (
        "block_src", "block_ngb", "block_T", "block_is_self",
        "block_dist", "block_offsets", "block_shapes",
    )]
    for k in mat_keys:
        g.create_dataset(k, data=col[k], chunks=chunk,
                         compression="gzip", compression_opts=_GZIP_LEVEL)


def write_molecular_material(
    h5_grp: h5py.Group,
    mid: str,
    payload: dict,
) -> None:
    """Write one molecular material with full (nao × nao) matrices."""
    g = h5_grp.create_group(mid)

    g.create_dataset("atomic_numbers", data=payload["atomic_numbers"].astype(np.int8))
    g.create_dataset("geometry_bohr",  data=payload["geometry_bohr"].astype(np.float32))

    g.attrs["charge"] = int(payload.get("charge", 0))
    _write_attrs(g, payload, (
        "energy_cp2k_Ha", "energy_dft_Ha",
        "bandgap_pbe", "bandgap_hse", "bandgap_cp2k",
    ))

    uks = "F_a" in payload
    mat_keys = ("H", "S", "F_a", "F_b", "P_a", "P_b") if uks else ("H", "S", "F", "P")
    for k in mat_keys:
        if k in payload:
            g.create_dataset(k, data=payload[k].astype(np.float32))

    H = payload.get("H")
    if H is not None:
        if uks:
            if "F_a" in payload:
                g.create_dataset("F_H_a", data=(payload["F_a"] - H).astype(np.float32))
            if "F_b" in payload:
                g.create_dataset("F_H_b", data=(payload["F_b"] - H).astype(np.float32))
        elif "F" in payload:
            g.create_dataset("F_H", data=(payload["F"] - H).astype(np.float32))


# ── Per-material loading helpers ──────────────────────────────────────────────

def _load_periodic(
    npz_path: Path,
    method: str,
    norb_z: Dict[int, int],
    topk: int,
    use_dist: bool,
) -> Tuple[dict, dict]:
    """Load periodic NPZ → (payload, columnar_block_arrays)."""
    data = read_periodic(npz_path)
    rspace = data["rspace"]

    if rspace:
        blocks = extract_rspace_blocks(data, norb_z, method=method, topk=topk, use_dist=use_dist)
    else:
        blocks = extract_kspace_blocks(data, norb_z, method=method)

    out_cp2k = npz_path.parent / "out.cp2k"
    m_charges, h_charges = parse_cp2k_charges(out_cp2k)
    n_atoms = len(data["atomic_numbers"])
    if m_charges is not None and len(m_charges) != n_atoms:
        m_charges = None
    if h_charges is not None and len(h_charges) != n_atoms:
        h_charges = None

    payload = {
        "atomic_numbers": data["atomic_numbers"],
        "geometry_bohr":  data["geometry_bohr"],
        "cell_bohr":      data["cell_bohr"],
        "pbc":            data["pbc"],
        "rspace":         rspace,
        "charge":         data.get("charge", 0),
        "net_spin":       data.get("net_spin", 0),
        "energy_cp2k_Ha": data.get("energy_cp2k_Ha"),
        "bandgap_pbesol": data.get("bandgap_pbe"),
        "bandgap_hse":    data.get("bandgap_hse"),
        "bandgap_cp2k":   data.get("bandgap_grid") or data.get("bandgap_cp2k"),
        "m_charges":      m_charges,
        "h_charges":      h_charges,
    }
    col = _blocks_to_columnar(blocks, rspace=rspace) if blocks else {}
    return payload, col


def _load_molecular(npz_path: Path, method: str) -> dict:
    """Load molecular NPZ; apply orbital permutation per atom-pair block; return payload for HDF5."""
    from .postproc import norb_by_z as _norb_by_z

    data = read_molecular(npz_path)
    elem = data["atomic_numbers"]
    permute = _get_permute_fn(method)
    norb_z = _norb_by_z(method)

    atom_ranges, offset = [], 0
    for z in elem:
        norb = norb_z[int(z)]
        atom_ranges.append((offset, offset + norb))
        offset += norb

    nao_from_z = atom_ranges[-1][1]
    nao_from_mat = data["H"].shape[0]
    assert nao_from_z == nao_from_mat, (
        f"orbital count mismatch: norb_by_z sums to {nao_from_z} "
        f"but matrix shape is ({nao_from_mat}, {nao_from_mat})"
    )

    def _permute_full(mat):
        dense = np.array(mat, dtype=np.float64)
        out = np.empty_like(dense)
        for i, (rs, re) in enumerate(atom_ranges):
            for j, (cs, ce) in enumerate(atom_ranges):
                out[rs:re, cs:ce] = permute(dense[rs:re, cs:ce], elem[i], elem[j])
        return out

    uks = data.get("unrestricted", False)
    mat_keys = ("H", "S", "F_a", "F_b", "P_a", "P_b") if uks else ("H", "S", "F", "P")
    for k in mat_keys:
        if k in data:
            data[k] = _permute_full(data[k])

    return {
        "atomic_numbers": elem,
        "geometry_bohr":  data["geometry_bohr"],
        "charge":         data.get("charge", 0),
        "energy_cp2k_Ha": data.get("energy_cp2k_Ha"),
        "energy_dft_Ha":  data.get("energy_dft_Ha"),
        **{k: data[k] for k in mat_keys if k in data},
    }



# ── Parallel loading worker ───────────────────────────────────────────────────

def _worker_periodic(args: tuple) -> tuple:
    mid, npz_path, method, norb_z, topk, use_dist = args
    try:
        payload, col = _load_periodic(npz_path, method, norb_z, topk, use_dist)
        return mid, payload, col, None
    except Exception as e:
        return mid, None, None, str(e)


# ── Unified split writer ──────────────────────────────────────────────────────

def write_split(
    h5_file: h5py.File,
    split_name: str,
    material_ids: Iterable[str],
    id_to_npz: Dict[str, Path],
    method: str,
    norb_z: Optional[Dict[int, int]] = None,
    topk: int = 32,
    n_workers: int = 1,
    use_dist: bool = False,
) -> int:
    """Write one train/val/test split to HDF5.

    Periodic methods use columnar block format; molecular methods write full dense matrices.
    """
    split_grp = h5_file.require_group(split_name)
    valid_ids = [mid for mid in material_ids if mid in id_to_npz]
    count = errors = 0

    mcfg = METHODS[method]

    if mcfg.periodic:
        # ── Periodic path ─────────────────────────────────────────────────────
        if n_workers > 1:
            args_list = [
                (mid, id_to_npz[mid], method, norb_z, topk, use_dist)
                for mid in valid_ids
            ]
            with Pool(n_workers) as pool:
                for mid, payload, col, err in tqdm(
                    pool.imap(_worker_periodic, args_list),
                    total=len(args_list), desc=f"[{split_name}]"
                ):
                    if err:
                        print(f"[WARN] {mid}: {err}")
                        errors += 1
                    else:
                        write_periodic_material(split_grp, mid, payload, col)
                        count += 1
        else:
            for mid in tqdm(valid_ids, desc=f"[{split_name}]"):
                try:
                    payload, col = _load_periodic(
                        id_to_npz[mid], method, norb_z, topk, use_dist
                    )
                    write_periodic_material(split_grp, mid, payload, col)
                    count += 1
                except Exception as e:
                    print(f"[WARN] Skipping {mid}: {e}")
                    errors += 1

    else:
        # ── Molecular path ────────────────────────────────────────────────────
        for mid in tqdm(valid_ids, desc=f"[{split_name}]"):
            try:
                payload = _load_molecular(id_to_npz[mid], method)
                write_molecular_material(split_grp, mid, payload)
                count += 1
            except Exception as e:
                print(f"[WARN] Skipping {mid}: {e}")
                errors += 1

    if errors:
        print(f"[{split_name}] {errors} entries skipped due to errors.")
    return count

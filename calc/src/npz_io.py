"""Stacked-COO NPZ intermediate format for CP2K matrix files.

Periodic R-space (rspace=True):
  T_vectors    int16  (nT, 3)
  {K}_tidx     int16  (nnz,)   ← T-vector index
  {K}_row      int16  (nnz,)
  {K}_col      int16  (nnz,)
  {K}_data     float32 (nnz,)
  for K in (H, S, F, P) [RKS] or (H, S, F_a, F_b, P_a, P_b) [UKS]

Periodic K-space (rspace=False):
  K_vectors    float32 (nK, 3)
  {K}_kstack   complex64 (nK, nao, nao)

Molecular (non-periodic):
  {K}          float32 (nao, nao)  for K in (H, S, F, P) or (H, S, F_a, F_b, P_a, P_b)

All formats also include:
  atomic_numbers  int16  (n_atoms,)
  geometry_bohr   float32 (n_atoms, 3)
  charge          int8
  uks             bool
  [periodic: cell_bohr float32 (3,3), pbc bool (3,), nao int32, net_spin int8, rspace bool]
  [optional scalars: energy_cp2k_Ha, bandgap_pbe, bandgap_hse, bandgap_grid, vbm_grid, cbm_grid]
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

_RKS_KEYS = ("H", "S", "F", "P")
_UKS_KEYS = ("H", "S", "F_a", "F_b", "P_a", "P_b")
_OPTIONAL_SCALARS = (
    "energy_cp2k_Ha", "energy_dft_Ha",
    "bandgap_pbe", "bandgap_hse", "bandgap_grid", "vbm_grid", "cbm_grid",
)


# ── Writers ───────────────────────────────────────────────────────────────────

def write_periodic(path: str | Path, data: dict) -> None:
    """Write periodic (PBC) matrices to a compressed NPZ file."""
    rspace = data.get("rspace", True)
    uks = data.get("unrestricted", False)
    mat_keys = _UKS_KEYS if uks else _RKS_KEYS

    arrays: dict = {
        "atomic_numbers": data["atomic_numbers"].astype(np.int16),
        "geometry_bohr":  data["geometry_bohr"].astype(np.float32),
        "cell_bohr":      data["cell_bohr"].astype(np.float32),
        "pbc":            np.array(data["pbc"], dtype=np.bool_),
        "nao":            np.int32(data.get("nao", 0)),
        "rspace":         np.bool_(rspace),
        "uks":            np.bool_(uks),
        "charge":         np.int8(data.get("charge", 0)),
        "net_spin":       np.int8(data.get("net_spin", 0)),
    }
    _pack_optional_scalars(arrays, data)

    matrices = data["matrices"]
    if rspace:
        T_list = list(matrices.keys())
        arrays["T_vectors"] = np.array(T_list, dtype=np.int16)
        for k in mat_keys:
            _pack_coo_key(arrays, matrices, T_list, k)
    else:
        K_list = list(matrices.keys())
        arrays["K_vectors"] = np.array(K_list, dtype=np.float32)
        nao = int(arrays["nao"])
        nK = len(K_list)
        first = matrices[K_list[0]]
        for k in mat_keys:
            if k not in first:
                continue
            dtype = np.complex64 if np.iscomplexobj(first[k]) else np.float32
            stacked = np.empty((nK, nao, nao), dtype=dtype)
            for ki, kv in enumerate(K_list):
                stacked[ki] = matrices[kv][k]
            arrays[f"{k}_kstack"] = stacked

    np.savez_compressed(path, **arrays)


def write_molecular(path: str | Path, data: dict) -> None:
    """Write non-periodic (molecular) matrices to a compressed NPZ file."""
    uks = data.get("unrestricted", False)
    mat_keys = _UKS_KEYS if uks else _RKS_KEYS

    arrays: dict = {
        "atomic_numbers": data["atomic_numbers"].astype(np.int16),
        "geometry_bohr":  data["geometry_bohr"].astype(np.float32),
        "uks":            np.bool_(uks),
        "charge":         np.int8(data.get("charge", 0)),
    }
    _pack_optional_scalars(arrays, data)
    for k in mat_keys:
        if data.get(k) is not None:
            arrays[k] = np.asarray(data[k], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _pack_coo_key(arrays: dict, matrices: dict, T_list: list, k: str) -> None:
    """Accumulate COO entries for matrix type k across all T-vectors."""
    tidx_parts, row_parts, col_parts, data_parts = [], [], [], []
    for ti, T in enumerate(T_list):
        m = matrices[T].get(k)
        if m is None:
            continue
        if hasattr(m, "tocoo"):
            coo = m.tocoo()
            rows, cols, vals = coo.row, coo.col, coo.data
        else:
            arr = np.asarray(m)
            nz = np.nonzero(arr)
            rows, cols, vals = nz[0], nz[1], arr[nz]
        if len(vals) == 0:
            continue
        n = len(vals)
        tidx_parts.append(np.full(n, ti, dtype=np.int16))
        row_parts.append(rows.astype(np.int16))
        col_parts.append(cols.astype(np.int16))
        data_parts.append(vals.astype(np.float32))
    if tidx_parts:
        arrays[f"{k}_tidx"] = np.concatenate(tidx_parts)
        arrays[f"{k}_row"]  = np.concatenate(row_parts)
        arrays[f"{k}_col"]  = np.concatenate(col_parts)
        arrays[f"{k}_data"] = np.concatenate(data_parts)


def _pack_optional_scalars(arrays: dict, data: dict) -> None:
    for key in _OPTIONAL_SCALARS:
        val = data.get(key)
        if val is not None:
            arrays[key] = np.float64(val)


# ── Readers ───────────────────────────────────────────────────────────────────

def read_periodic(path: str | Path) -> dict:
    """Read periodic NPZ. Returns a dict in the same format as postproc output."""
    with np.load(path, allow_pickle=False) as f:
        files = set(f.files)
        rspace = bool(f["rspace"])
        uks = bool(f["uks"])
        nao = int(f["nao"])
        mat_keys = _UKS_KEYS if uks else _RKS_KEYS

        data: dict = {
            "atomic_numbers": f["atomic_numbers"].astype(np.int64),
            "geometry_bohr":  f["geometry_bohr"].astype(np.float64),
            "cell_bohr":      f["cell_bohr"].astype(np.float64),
            "pbc":            f["pbc"].tolist(),
            "nao":            nao,
            "rspace":         rspace,
            "unrestricted":   uks,
            "charge":         int(f["charge"]),
            "net_spin":       int(f["net_spin"]),
        }
        for key in _OPTIONAL_SCALARS:
            if key in files:
                data[key] = float(f[key])

        if rspace:
            T_arr = f["T_vectors"]  # (nT, 3)
            T_list = [tuple(int(x) for x in row) for row in T_arr]
            matrices: dict = {T: {} for T in T_list}
            for k in mat_keys:
                if f"{k}_tidx" not in files:
                    continue
                tidx = f[f"{k}_tidx"].astype(np.int32)
                rows = f[f"{k}_row"].astype(np.int32)
                cols = f[f"{k}_col"].astype(np.int32)
                vals = f[f"{k}_data"].astype(np.float64)
                for ti, T in enumerate(T_list):
                    mask = tidx == ti
                    if mask.any():
                        matrices[T][k] = csr_matrix(
                            (vals[mask], (rows[mask], cols[mask])),
                            shape=(nao, nao),
                        )
            data["matrices"] = matrices
        else:
            K_arr = f["K_vectors"]  # (nK, 3)
            K_list = [tuple(float(x) for x in row) for row in K_arr]
            matrices = {}
            for ki, kv in enumerate(K_list):
                mats: dict = {}
                for k in mat_keys:
                    key_stack = f"{k}_kstack"
                    if key_stack in files:
                        mats[k] = f[key_stack][ki].astype(np.complex128)
                matrices[kv] = mats
            data["matrices"] = matrices
            data["kpoints_2pi_bohr"] = K_arr.astype(np.float64)
    return data


def read_molecular(path: str | Path) -> dict:
    """Read molecular NPZ. Returns a dict with dense matrices."""
    with np.load(path, allow_pickle=False) as f:
        files = set(f.files)
        uks = bool(f["uks"]) if "uks" in files else False
        mat_keys = _UKS_KEYS if uks else _RKS_KEYS
        data: dict = {
            "atomic_numbers": f["atomic_numbers"].astype(np.int64),
            "geometry_bohr":  f["geometry_bohr"].astype(np.float64),
            "unrestricted":   uks,
            "charge":         int(f["charge"]),
        }
        for key in ("energy_cp2k_Ha", "energy_dft_Ha"):
            if key in files:
                data[key] = float(f[key])
        for k in mat_keys:
            if k in files:
                data[k] = f[k].astype(np.float64)
    return data

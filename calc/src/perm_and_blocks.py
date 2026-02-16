import numpy as np
import json
from typing import Dict, List, Tuple, Any

from .config import PBE_PATH

def read_xyz(xyzfile_path):
    n_atoms = np.loadtxt(xyzfile_path, max_rows=1, dtype=int)[()]
    coordinates = (
        np.loadtxt(
            xyzfile_path, skiprows=2, usecols=[1, 2, 3], max_rows=n_atoms
        ).reshape((n_atoms, 3))
        * 1.8897259886 # angstrom to bohr
    )
    return coordinates

def build_atom_ranges(norb_z: dict, atoms: list) -> List[Tuple[int, int]]:
    orbital_idx = 0
    atom_ranges = []

    for atom in atoms:
        n_orb = norb_z[atom['atomnum']]
        atom_ranges.append((orbital_idx, orbital_idx + n_orb))
        orbital_idx += n_orb
    
    return atom_ranges

def get_block(matrix, src, ngb, atom_ranges):
    """
    Extract the block corresponding to interactions between two atoms.

    Works with both dense and sparse matrices.
    Returns dense block in both cases.
    """
    src_start, src_end = atom_ranges[src - 1]  # atoms are 1-indexed
    ngb_start, ngb_end = atom_ranges[ngb - 1]

    block = matrix[src_start:src_end, ngb_start:ngb_end]

    # Convert sparse block to dense for downstream processing
    if hasattr(block, 'toarray'):
        return block.toarray()
    return block


def compute_block_norm_squared_sparse(matrix_csr, src, ngb, atom_ranges):
    """
    Compute squared Frobenius norm of a block from sparse matrix efficiently.

    This avoids materializing the dense block, computing directly from nonzeros.

    Args:
        matrix_csr: Sparse matrix in CSR format
        src: Source atom index (1-indexed)
        ngb: Neighbor atom index (1-indexed)
        atom_ranges: List of (start, end) tuples for each atom's orbital range

    Returns:
        Squared Frobenius norm of the block
    """
    src_start, src_end = atom_ranges[src - 1]
    ngb_start, ngb_end = atom_ranges[ngb - 1]

    # Extract the sparse block
    block = matrix_csr[src_start:src_end, ngb_start:ngb_end]

    if hasattr(block, 'data'):
        # Sparse matrix: sum of squares of nonzero values
        return float(np.sum(block.data ** 2))
    else:
        # Dense fallback
        return float(np.sum(block ** 2))

# ============================================================
# Orbital permutation utilities (export _permute_block)
# for crystals
# ============================================================

def perm_map_H(): return np.array([0, 1])
def perm_map_He(): return np.array([0])
def perm_map_nsp(): return np.array([0, 3, 2, 1])
def perm_map_transition_metals_lanthanides(): return np.array([5, 8, 7, 6, 4, 3, 2, 1, 0])
def perm_map_else(): return np.array([0, 3, 2, 1, 8, 7, 6, 5, 4])

def is_H(z): return z == 1
def is_He(z): return z == 2
def is_row_2_el(z): return 3 <= z <= 9
def is_group_1_el(z): return z in (11, 19, 37, 55)
def is_zncdhgtlpbbi_mg(z): return z in (12, 30, 48) or (80 <= z <= 83)
def is_transition_metals_lanthanides(z):
    return (21 <= z <= 29) or (39 <= z <= 47) or (57 <= z <= 79)

def _atom_perm_for(z, norb):
    if is_H(z):
        pm = perm_map_H()
    elif is_row_2_el(z) or is_group_1_el(z) or is_zncdhgtlpbbi_mg(z):
        pm = perm_map_nsp()
    elif is_transition_metals_lanthanides(z):
        pm = perm_map_transition_metals_lanthanides()
    elif is_He(z):
        pm = perm_map_He()
    else:
        pm = perm_map_else()
    if pm.size != int(norb):
        return np.arange(int(norb))
    return pm

def _permute_block(M, z_src, z_nbr):
    """Apply per-atom permutation to a pair block (rows: src, cols: nbr)."""
    Pi = _atom_perm_for(int(z_src), M.shape[0])
    Pj = _atom_perm_for(int(z_nbr), M.shape[1])
    return M[np.ix_(Pi, Pj)]


# ============================================================
# PBE orbital permutation utilities
# Reorders orbitals within each shell from m=-l,...,l to m=l,...,-l
# ============================================================

_pbe_basis_cache = None

def _load_pbe_basis() -> Dict:
    """Load and cache PBE basis info from JSON."""
    global _pbe_basis_cache
    if _pbe_basis_cache is None:
        with open(PBE_PATH) as f:
            _pbe_basis_cache = json.load(f)
    return _pbe_basis_cache


# Shell sizes: s=1, p=3, d=5, f=7
_SHELL_SIZES = {"s": 1, "p": 3, "d": 5, "f": 7}
_SHELL_ORDER = ["s", "p", "d", "f"]


def _atom_perm_pbe(z: int) -> np.ndarray:
    """
    Build permutation array for a given atomic number using PBE basis.

    Reverses orbital order within each shell (m=-l,...,l → m=l,...,-l).
    Shell order (s, p, d, f) is preserved.
    """
    basis = _load_pbe_basis()
    z_str = str(z)

    if z_str not in basis:
        raise KeyError(f"Atomic number {z} not found in PBE basis")

    shells = basis[z_str].get("shells", {})
    total = basis[z_str]["total"]

    perm = []
    offset = 0

    for shell_name in _SHELL_ORDER:
        if shell_name not in shells:
            continue

        n_orb_in_shell = shells[shell_name]
        shell_size = _SHELL_SIZES[shell_name]

        # Number of contracted functions for this shell type
        n_contracted = n_orb_in_shell // shell_size

        for _ in range(n_contracted):
            # Reverse the order within this shell block
            shell_indices = list(range(offset + shell_size - 1, offset - 1, -1))
            perm.extend(shell_indices)
            offset += shell_size

    if len(perm) != total:
        raise ValueError(f"Permutation length {len(perm)} doesn't match total orbitals {total} for Z={z}")

    return np.array(perm, dtype=int)


def _permute_block_pbe(M, z_src, z_nbr):
    """Apply PBE orbital permutation to a pair block (rows: src, cols: nbr)."""
    Pi = _atom_perm_pbe(int(z_src))
    Pj = _atom_perm_pbe(int(z_nbr))
    return M[np.ix_(Pi, Pj)]


## For QM9 molecules ##

def get_perm_map(element_numbers):
    """
    Permutation map for converting the matrices obtained from `tblite`
    GFN-xTB (closed shell) to the convention for qcore.

    Input (tblite) convention:
     - H: [1s, 2s]
     - C,N,O,F: [2s, 2py, 2pz, 2px]
    Output (qcore) convention:
     - H: [1s, 2s] (no change)
     - C,N,O,F: [2s, 2px, 2pz, 2py]
    """
    n_so = 0
    for el in element_numbers:
        if el == 1:
            n_so += 2
        else:
            n_so += 4

    perm_map = np.zeros(n_so, dtype=int)
    idx = 0
    for el in element_numbers:
        if el == 1:
            pmap = [idx, idx + 1]
            perm_map[idx : idx + 2] = pmap
            idx += 2
        else:
            # [s, py, pz, px] -> [s, px, pz, py]
            pmap = [idx, idx + 3, idx + 2, idx + 1]
            perm_map[idx : idx + 4] = pmap
            idx += 4
    return perm_map

def apply_perm_map(mat, perm_map):
    ndim = mat.ndim
    if ndim == 1:
        matp = mat[perm_map]
    elif ndim == 2:
        matp = mat[perm_map, :]
        matp = matp[:, perm_map]
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")
    return matp

def build_atom_blocks(basis: dict, atoms: list) -> dict:
    """Build orbital index blocks from parsed data."""
    blocks = {}
    orbital_idx = 0
    for atom in atoms:
        start = orbital_idx
        end = orbital_idx + basis[atom['element']]
        blocks[f"{atom['element']}_{atom['idx']}"] = (start, end)
        orbital_idx = end
    return blocks
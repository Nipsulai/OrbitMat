"""Parsers for CP2K output files: log, binary CSR matrices, and MOLog eigenvalues."""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .config import OUTPUT, ELEM_PATH, METHODS

ANG_TO_BOHR = 1.8897259886
HA_TO_EV    = 27.211386245
TEMP_SAVE_DIR = "matrices"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Atom:
    idx: int
    element: str
    atomnum: int
    coord_bohr: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class BandgapResult:
    bandgap_eV: float
    vbm_eV: float
    cbm_eV: float
    is_direct: bool
    vbm_kpoint: int
    cbm_kpoint: int
    is_metal: bool = False


# ── Element lookup ────────────────────────────────────────────────────────────

_elem_symb_atomnumber_dict = None


def get_elem_symb_atomnumber_dict():
    """Lazily load element symbol → atomic number mapping."""
    global _elem_symb_atomnumber_dict
    if _elem_symb_atomnumber_dict is None:
        _elem_symb_atomnumber_dict = dict(
            zip(np.loadtxt(str(ELEM_PATH), dtype=str), np.arange(118) + 1)
        )
    return _elem_symb_atomnumber_dict


# ── CP2K log parser ───────────────────────────────────────────────────────────

def parse_cp2k_output(workdir, method, rspace):
    """Parse a CP2K output log and return geometry, k/T-vectors, and energy."""
    nao = 0
    atoms = []
    coords_bohr = []
    in_coords = False
    energy_Ha = None
    T_vectors = None
    K_vectors = None
    K_weights = None
    lattice_vecs_bohr = None

    elem_symb_atomnumber_dict = get_elem_symb_atomnumber_dict()

    with open(f"{workdir}/{OUTPUT}", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Number of orbital functions:' in line:
            nao = int(line.split()[-1])

        if 'CELL| Vector a [angstrom]:' in line:
            parts_a = line.split()
            vec_a = [float(parts_a[4]), float(parts_a[5]), float(parts_a[6])]
            if i + 1 < len(lines) and 'CELL| Vector b [angstrom]:' in lines[i + 1]:
                parts_b = lines[i + 1].split()
                vec_b = [float(parts_b[4]), float(parts_b[5]), float(parts_b[6])]
            if i + 2 < len(lines) and 'CELL| Vector c [angstrom]:' in lines[i + 2]:
                parts_c = lines[i + 2].split()
                vec_c = [float(parts_c[4]), float(parts_c[5]), float(parts_c[6])]
            lattice_vecs_bohr = np.array([vec_a, vec_b, vec_c], dtype=np.float32) * ANG_TO_BOHR

        if 'CSR writ' in line and 'periodic images' in line:
            match = re.search(r'(\d+)\s+periodic images', line)
            if match:
                nT = int(match.group(1))
                T_list = []
                for j in range(1, nT + 1):
                    if i + j + 1 >= len(lines):
                        break
                    parts = lines[i + j + 1].strip().split()
                    if len(parts) >= 4:
                        T_list.append([int(parts[1]), int(parts[2]), int(parts[3])])
                if len(T_list) != nT:
                    warnings.warn(
                        f"Expected {nT} T-vectors but parsed {len(T_list)}. "
                        "File may be truncated or malformed."
                    )
                T_vectors = np.array(T_list, dtype=np.int32)

        if 'BRILLOUIN| List of Kpoints' in line:
            nK = int(line.split()[-1])
            K_list, W_list = [], []
            for j in range(1, nK + 1):
                if i + j + 1 >= len(lines):
                    break
                data_line = lines[i + j + 1]
                if 'BRILLOUIN|' not in data_line:
                    break
                parts = data_line.strip().split()
                if len(parts) >= 6:
                    W_list.append(float(parts[2]))
                    K_list.append([float(parts[3]), float(parts[4]), float(parts[5])])
            if len(K_list) != nK:
                warnings.warn(
                    f"Expected {nK} K-points but parsed {len(K_list)}. "
                    "File may be truncated or malformed."
                )
            K_vectors = np.array(K_list, dtype=np.float64)
            K_weights = np.array(W_list, dtype=np.float64)

        if "MODULE QUICKSTEP: ATOMIC COORDINATES" in line:
            in_coords = True
            continue

        if "ENERGY| Total FORCE_EVAL" in line:
            energy_Ha = float(line.split()[-1])

        if in_coords:
            if "Atom Kind Element" in line:
                continue
            if not line.strip() or line.strip().startswith('---'):
                if atoms:
                    in_coords = False
                continue
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    X = float(parts[4]) * ANG_TO_BOHR
                    Y = float(parts[5]) * ANG_TO_BOHR
                    Z = float(parts[6]) * ANG_TO_BOHR
                    atoms.append(Atom(
                        idx=int(parts[0]),
                        element=parts[2],
                        atomnum=elem_symb_atomnumber_dict[parts[2]],
                        coord_bohr=np.array([X, Y, Z]),
                    ))
                    coords_bohr.append([X, Y, Z])
                except (ValueError, KeyError):
                    pass
            elif atoms:
                in_coords = False

    if nao == 0:
        raise ValueError("Could not find 'Number of orbital functions'")
    if energy_Ha is None:
        raise ValueError("Could not parse energy")
    if len(atoms) == 0:
        raise ValueError("Could not parse atoms")
    if len(coords_bohr) != len(atoms):
        raise ValueError("Mismatch between atoms and coordinates")

    if METHODS[method].periodic:
        if rspace and T_vectors is None:
            raise ValueError("Could not find T_vectors in output file")
        if not rspace and K_vectors is None:
            raise ValueError("Could not find K_vectors in output file")

    return nao, atoms, np.array(coords_bohr), energy_Ha, T_vectors, K_vectors, K_weights, lattice_vecs_bohr


# ── UKS detection ─────────────────────────────────────────────────────────────

def _detect_uks(work_dir: Path, project: str, periodic: bool, rspace: bool) -> bool:
    """Return True if CP2K produced spin-2 KS files (UKS calculation)."""
    mdir = work_dir / TEMP_SAVE_DIR
    if periodic:
        space = "R" if rspace else "K"
        return (mdir / f"{project}-KS_SPIN_2_{space}_1-1_0.csr").exists()
    else:
        return (mdir / f"{project}-KS_SPIN_2-1_0.csr").exists()


# ── Binary CSR matrix parser ──────────────────────────────────────────────────

def parse_matrix(work_dir, method, matrix_type, nao, binary=True, TK_idx=None,
                 keep_sparse=True, rspace=False, cutoff=1e-12, spin=1):
    """
    Parse a CP2K binary (or text) CSR matrix file.

    Args:
        keep_sparse: Return scipy.sparse.csr_matrix when True, dense ndarray otherwise.
        rspace:      R-space files are real float64 (24 B/record);
                     K-space files are complex128 (32 B/record).
        cutoff:      Zero out elements with |val| < cutoff before building the sparse
                     matrix (eliminates numerical noise). Set to 0 to disable.
        spin:        Spin channel — 1 (alpha/closed-shell) or 2 (beta, UKS only).
    """
    project = METHODS[method].project_name
    if TK_idx is not None:
        space = "R" if rspace else "K"
        filepath = f"{work_dir}/{TEMP_SAVE_DIR}/{project}-{matrix_type}_SPIN_{spin}_{space}_{TK_idx+1}-1_0.csr"
    else:
        filepath = f"{work_dir}/{TEMP_SAVE_DIR}/{project}-{matrix_type}_SPIN_{spin}-1_0.csr"

    if binary:
        with open(filepath, "rb") as f:
            raw_data = np.fromfile(f, dtype=np.uint8)

        if rspace:
            # R-space: int32 marker | int32 row | int32 col | float64 val | int32 marker  (24 B)
            record_size = 24
            dtype = np.dtype([('marker1', np.int32), ('row', np.int32),
                               ('col', np.int32), ('value', np.float64), ('marker2', np.int32)])
            empty_mat = csr_matrix((nao, nao))
        else:
            # K-space: int32 marker | int32 row | int32 col | complex128 val | int32 marker  (32 B)
            record_size = 32
            dtype = np.dtype([('marker1', np.int32), ('row', np.int32),
                               ('col', np.int32), ('value', np.complex128), ('marker2', np.int32)])
            empty_mat = csr_matrix((nao, nao), dtype=np.complex64)

        n_records = len(raw_data) // record_size
        if n_records == 0:
            return empty_mat if keep_sparse else empty_mat.toarray()

        records = np.frombuffer(raw_data[:n_records * record_size], dtype=dtype)
        rows = records['row'].astype(np.int32) - 1
        cols = records['col'].astype(np.int32) - 1
        data = records['value'].astype(np.float32 if rspace else np.complex64)

        if cutoff > 0:
            mask = np.abs(data) >= cutoff
            data, rows, cols = data[mask], rows[mask], cols[mask]

        mat = csr_matrix((data, (rows, cols)), shape=(nao, nao))
        return mat if keep_sparse else mat.toarray()
    else:
        r, c, data = np.loadtxt(filepath, unpack=True)
        if cutoff > 0:
            mask = np.abs(data) >= cutoff
            r, c, data = r[mask], c[mask], data[mask]
        mat = csr_matrix((data, (np.int32(r - 1), np.int32(c - 1))), shape=(nao, nao))
        return mat if keep_sparse else mat.toarray()


# ── MOLog eigenvalue parser ───────────────────────────────────────────────────

def parse_molog(filepath, occ_threshold: float = 0.5) -> BandgapResult:
    """Parse a CP2K .MOLog file and return a BandgapResult."""
    with open(filepath, 'r') as f:
        content = f.read()

    kpoint_pattern   = r'MO\| ((?:ALPHA |BETA )?)EIGENVALUES AND OCCUPATION NUMBERS FOR K POINT (\d+)'
    eigenvalue_pattern = r'MO\|\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'

    kpoint_matches = list(re.finditer(kpoint_pattern, content))
    if not kpoint_matches:
        raise ValueError("No k-point data found in file")

    kpoint_data = {}  # (spin_label, kpoint_idx) -> {'vbm': float, 'cbm': float|None}

    for i, kpoint_match in enumerate(kpoint_matches):
        spin_label = kpoint_match.group(1).strip() or "RKS"
        kpoint_idx = int(kpoint_match.group(2))
        start_pos  = kpoint_match.end()
        end_pos    = kpoint_matches[i + 1].start() if i + 1 < len(kpoint_matches) else len(content)
        section    = content[start_pos:end_pos]

        eigenvalues = [
            {'index': int(m.group(1)), 'eigval_eV': float(m.group(3)), 'occupation': float(m.group(4))}
            for m in re.finditer(eigenvalue_pattern, section)
        ]
        if not eigenvalues:
            continue

        occupied   = [e for e in eigenvalues if e['occupation'] >= occ_threshold]
        unoccupied = [e for e in eigenvalues if e['occupation'] < occ_threshold]

        if occupied and unoccupied:
            vbm = max(occupied,   key=lambda x: x['eigval_eV'])['eigval_eV']
            cbm = min(unoccupied, key=lambda x: x['eigval_eV'])['eigval_eV']
            kpoint_data[(spin_label, kpoint_idx)] = {'vbm': vbm, 'cbm': cbm}
        elif occupied:
            vbm = max(occupied, key=lambda x: x['eigval_eV'])['eigval_eV']
            kpoint_data[(spin_label, kpoint_idx)] = {'vbm': vbm, 'cbm': None}

    if not kpoint_data:
        raise ValueError("Could not extract eigenvalue data from file")

    has_cbm = any(d['cbm'] is not None for d in kpoint_data.values())

    if not has_cbm:
        global_vbm = max(d['vbm'] for d in kpoint_data.values())
        vbm_kpoint = max(kpoint_data, key=lambda k: kpoint_data[k]['vbm'])[1]
        return BandgapResult(
            bandgap_eV=0.0, vbm_eV=global_vbm, cbm_eV=global_vbm,
            is_direct=True, vbm_kpoint=vbm_kpoint, cbm_kpoint=vbm_kpoint, is_metal=True,
        )

    global_vbm, vbm_kpoint = float('-inf'), None
    for key, d in kpoint_data.items():
        if d['vbm'] > global_vbm:
            global_vbm, vbm_kpoint = d['vbm'], key[1]

    global_cbm, cbm_kpoint = float('inf'), None
    for key, d in kpoint_data.items():
        if d['cbm'] is not None and d['cbm'] < global_cbm:
            global_cbm, cbm_kpoint = d['cbm'], key[1]

    bandgap = global_cbm - global_vbm
    return BandgapResult(
        bandgap_eV=max(0.0, bandgap),
        vbm_eV=global_vbm,
        cbm_eV=global_cbm,
        is_direct=(vbm_kpoint == cbm_kpoint),
        vbm_kpoint=vbm_kpoint,
        cbm_kpoint=cbm_kpoint,
        is_metal=(bandgap <= 0),
    )

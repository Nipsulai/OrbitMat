from scipy.sparse import csr_matrix
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from dataclasses import dataclass
import json
import re
import pickle

from .config import OUTPUT, ELEM_PATH, XTB_PATH, PBE_PATH
from .perm_and_blocks import _permute_block, _permute_block_pbe, apply_perm_map, get_perm_map, build_atom_ranges, get_block, compute_block_norm_squared_sparse

ANG_TO_BOHR = 1.8897259886
TEMP_SAVE_DIR = "matrices"

@dataclass
class Atom:
    idx: int
    element: str
    atomnum: int
    coord_bohr: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)

_elem_symb_atomnumber_dict = None

def get_elem_symb_atomnumber_dict():
    """Lazily load element symbol to atomic number mapping."""
    global _elem_symb_atomnumber_dict
    if _elem_symb_atomnumber_dict is None:
        _elem_symb_atomnumber_dict = dict(
            zip(np.loadtxt(f"{ELEM_PATH}", dtype=str), np.arange(118) + 1)
        )
    return _elem_symb_atomnumber_dict

def norb_by_z(method):
    if method == "xtb":
        json_path = XTB_PATH
        with open(json_path) as f:
            return {int(z): norb for z, norb in json.load(f).items()}
    elif method == "pbe":
        json_path= PBE_PATH
        with open(json_path) as f:
            return {int(z): norb["total"] for z, norb in json.load(f).items()}
    

def parse_cp2k_output(workdir):
    nao = 0
    atoms = []
    coords_bohr= []
    in_coords = False
    energy_Ha = None
    T_vectors = None
    lattice_vecs_bohr = None

    # Load element mapping
    elem_symb_atomnumber_dict = get_elem_symb_atomnumber_dict()

    with open(f"{workdir}/{OUTPUT}", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Number of orbital functions:' in line:
            nao = int(line.split()[-1])

        if 'CELL| Vector a [angstrom]:' in line:
            # Parse lattice vectors a, b, c
            # Format: "CELL| Vector a [angstrom]:       4.587     0.000     0.000   |a| =     4.586627"
            parts_a = line.split()
            vec_a = [float(parts_a[4]), float(parts_a[5]), float(parts_a[6])]

            # Next two lines should be vectors b and c
            if i + 1 < len(lines) and 'CELL| Vector b [angstrom]:' in lines[i + 1]:
                parts_b = lines[i + 1].split()
                vec_b = [float(parts_b[4]), float(parts_b[5]), float(parts_b[6])]

            if i + 2 < len(lines) and 'CELL| Vector c [angstrom]:' in lines[i + 2]:
                parts_c = lines[i + 2].split()
                vec_c = [float(parts_c[4]), float(parts_c[5]), float(parts_c[6])]

            lattice_vecs_bohr = np.array([vec_a, vec_b, vec_c], dtype=np.float32)*ANG_TO_BOHR

        if 'CSR writ' in line and 'periodic images' in line:
            # Format can be:
            # "KS CSR write|2037 periodic images"
            # or malformed (e.g. missing 'e|')
            # "HCORE CSR writ2037 periodic images"
            # Extract the number robustly using regex
            match = re.search(r'(\d+)\s+periodic images', line)
            if match:
                nT = int(match.group(1))
                # Skip the header line "Number    X      Y      Z"
                T_list = []
                for j in range(1, nT + 1):
                    if i + j + 1 >= len(lines):
                        break
                    data_line = lines[i + j + 1]
                    parts = data_line.strip().split()
                    if len(parts) >= 4:
                        # Index is parts[0], X Y Z are parts[1:4]
                        T_list.append([int(parts[1]), int(parts[2]), int(parts[3])])

                # Validate that we parsed the expected number of T-vectors
                if len(T_list) != nT:
                    import warnings
                    warnings.warn(
                        f"Expected {nT} T-vectors but parsed {len(T_list)}. "
                        "File may be truncated or malformed."
                    )
                T_vectors = np.array(T_list, dtype=np.int32)

        if "MODULE QUICKSTEP: ATOMIC COORDINATES" in line:
            in_coords = True
            continue
    
        if "ENERGY| Total FORCE_EVAL" in line:
            energy_Ha = float(line.split()[-1])
        
        if in_coords:
            # Skip header lines
            if "Atom Kind Element" in line:
                continue

            # End coordinate parsing on blank line or separator
            if not line.strip() or line.strip().startswith('---'):
                if atoms:  # Only end if we've found atoms
                    in_coords = False
                continue

            parts = line.split()
            # Atom line starts with atom_index and has enough fields
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    X = float(parts[4]) * ANG_TO_BOHR
                    Y = float(parts[5]) * ANG_TO_BOHR
                    Z = float(parts[6]) * ANG_TO_BOHR

                    atoms.append(Atom(
                        idx=int(parts[0]),
                        element=parts[2],
                        atomnum=elem_symb_atomnumber_dict[parts[2]],
                        coord_bohr=np.array([X, Y, Z])
                    ))

                    coords_bohr.append([X, Y, Z])
                except (ValueError, KeyError):
                    # Skip malformed lines but continue parsing
                    pass
            elif atoms:
                # If we have atoms and hit a non-atom line, stop parsing
                in_coords = False

    if nao == 0:
        raise ValueError("Could not find 'Number of orbital functions'")
    if energy_Ha is None:
        raise ValueError("Could not parse energy")
    if len(atoms) == 0:
        raise ValueError("Could not parse atoms")
    if len(coords_bohr) != len(atoms):
        raise ValueError("Mismatch between atoms and coordinates")
    
    # Convert to numpy array in Bohr
    coords_bohr = np.array(coords_bohr)

    return nao, atoms, coords_bohr, energy_Ha, T_vectors, lattice_vecs_bohr

def parse_matrix(work_dir, method, matrix_type, nao, binary=True, T_idx=None, keep_sparse=True):
    """
    Parse matrix from file.

    Args:
        keep_sparse: If True, return scipy.sparse.csr_matrix. If False, return dense numpy array.
                     Defaults to True for better performance.
    """
    if T_idx is not None:
        # R-space matrix files are in "matrices" subdirectory
        # File naming: method-matrix_type_SPIN_1_R_T_idx-1_0.csr
        filepath = f"{work_dir}/{TEMP_SAVE_DIR}/{method}-{matrix_type}_SPIN_1_R_{T_idx+1}-1_0.csr"
    else:
        filepath = f"{work_dir}/{TEMP_SAVE_DIR}/{method}-{matrix_type}_SPIN_1-1_0.csr"
    if binary:
        with open(filepath, "rb") as f:
            raw_data = np.fromfile(f, dtype=np.uint8)

        # Each record is 24 bytes
        n_records = len(raw_data) // 24

        # Empty file
        if n_records == 0:
            return csr_matrix((nao, nao)) if keep_sparse else np.zeros((nao, nao))

        # Use structured dtype to parse without intermediate copies
        # Structure: int32 marker, int32 row, int32 col, float64 value, int32 marker (24 bytes)
        dtype = np.dtype([
            ('marker1', np.int32),
            ('row', np.int32),
            ('col', np.int32),
            ('value', np.float64),
            ('marker2', np.int32)
        ])

        # View raw bytes as structured array (zero-copy)
        records = np.frombuffer(raw_data[:n_records * 24], dtype=dtype)

        # Extract fields directly (creates views, minimal copying)
        rows = records['row'].astype(np.int32) - 1  # Convert to 0-indexed
        cols = records['col'].astype(np.int32) - 1  # Convert to 0-indexed
        data = records['value'].astype(np.float32)  # Convert to float32 for memory efficiency

        mat = csr_matrix((data, (rows, cols)), shape=(nao, nao))
        return mat if keep_sparse else mat.toarray()
    else:
        r, c, data = np.loadtxt(
            filepath,
            unpack=True,
        )
        mat = csr_matrix(
            (data, (np.int32(r - 1), np.int32(c - 1))),
            shape=(nao, nao)
        )
        return mat if keep_sparse else mat.toarray()

from dataclasses import dataclass
from typing import Optional


@dataclass
class BandgapResult:
    """Results from bandgap calculation."""
    bandgap_eV: float
    vbm_eV: float
    cbm_eV: float
    is_direct: bool
    vbm_kpoint: int
    cbm_kpoint: int
    is_metal: bool = False


def parse_molog(workdir, occ_threshold: float = 0.5) -> BandgapResult:
    filepath = f"{workdir}/{TEMP_SAVE_DIR}/eigenvalues-1_0.MOLog"
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match k-point blocks
    kpoint_pattern = r'MO\| EIGENVALUES AND OCCUPATION NUMBERS FOR K POINT (\d+)'
    # Pattern to match eigenvalue lines
    eigenvalue_pattern = r'MO\|\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'

    # Find all k-point sections
    kpoint_matches = list(re.finditer(kpoint_pattern, content))

    if not kpoint_matches:
        raise ValueError("No k-point data found in file")

    # Store VBM and CBM
    kpoint_data = {}  # kpoint_idx: {'vbm': value, 'cbm': value}

    for i, kpoint_match in enumerate(kpoint_matches):
        kpoint_idx = int(kpoint_match.group(1))

        # Find the end of this k-point section
        start_pos = kpoint_match.end()
        if i + 1 < len(kpoint_matches):
            end_pos = kpoint_matches[i + 1].start()
        else:
            end_pos = len(content)

        section = content[start_pos:end_pos]

        # Parse all eigenvalues in this section
        eigenvalues = []
        for match in re.finditer(eigenvalue_pattern, section):
            idx = int(match.group(1))
            eigval_au = float(match.group(2))
            eigval_eV = float(match.group(3))
            occupation = float(match.group(4))
            eigenvalues.append({
                'index': idx,
                'eigval_eV': eigval_eV,
                'occupation': occupation
            })

        if not eigenvalues:
            continue

        # Find VBM and CBM
        occupied = [e for e in eigenvalues if e['occupation'] >= occ_threshold]
        unoccupied = [e for e in eigenvalues if e['occupation'] < occ_threshold]

        if occupied and unoccupied:
            vbm = max(occupied, key=lambda x: x['eigval_eV'])['eigval_eV']
            cbm = min(unoccupied, key=lambda x: x['eigval_eV'])['eigval_eV']
            kpoint_data[kpoint_idx] = {'vbm': vbm, 'cbm': cbm}
        elif occupied and not unoccupied:
            # All states occupied, might be metallic or file doesn't include unoccupied states
            vbm = max(occupied, key=lambda x: x['eigval_eV'])['eigval_eV']
            kpoint_data[kpoint_idx] = {'vbm': vbm, 'cbm': None}

    if not kpoint_data:
        raise ValueError("Could not extract eigenvalue data from file")

    # Check if we have CBM data
    has_cbm = any(data['cbm'] is not None for data in kpoint_data.values())

    if not has_cbm:
        # All states are occupied,likely metallic or need more unoccupied states in calculation
        global_vbm = max(data['vbm'] for data in kpoint_data.values())
        vbm_kpoint = max(kpoint_data.keys(), key=lambda k: kpoint_data[k]['vbm'])
        return BandgapResult(
            bandgap_eV=0.0,
            vbm_eV=global_vbm,
            cbm_eV=global_vbm,  # No gap
            is_direct=True,
            vbm_kpoint=vbm_kpoint,
            cbm_kpoint=vbm_kpoint,
            is_metal=True
        )

    # Find global VBM
    vbm_kpoint = None
    global_vbm = float('-inf')
    for kpoint_idx, data in kpoint_data.items():
        if data['vbm'] > global_vbm:
            global_vbm = data['vbm']
            vbm_kpoint = kpoint_idx

    # Find global CBM
    cbm_kpoint = None
    global_cbm = float('inf')
    for kpoint_idx, data in kpoint_data.items():
        if data['cbm'] is not None and data['cbm'] < global_cbm:
            global_cbm = data['cbm']
            cbm_kpoint = kpoint_idx

    # Calculate bandgap
    bandgap = global_cbm - global_vbm

    # Check if direct
    is_direct = (vbm_kpoint == cbm_kpoint)

    is_metal = bandgap <= 0

    return BandgapResult(
        bandgap_eV=max(0.0, bandgap),
        vbm_eV=global_vbm,
        cbm_eV=global_cbm,
        is_direct=is_direct,
        vbm_kpoint=vbm_kpoint,
        cbm_kpoint=cbm_kpoint,
        is_metal=is_metal
    )


def postproc_matrices(workdir, method, sym, get_energy=True, cutoff=1e-32, out_name="matrices", compress=True, topk=32, bandgap_info=None):
    workdir = Path(workdir)
    nao, atoms, coords_bohr, energy_Ha, T_vectors, lattice_vecs_bohr = parse_cp2k_output(workdir)

    if T_vectors is None and method != "xyz":
        raise ValueError("Could not find T_vectors in output file")

    elem_numbers = np.array([a["atomnum"] for a in atoms])

    if method == "xyz":
        perm_map = get_perm_map(elem_numbers)
        F, S, H, P = (
            parse_matrix(workdir, method, mat, nao, binary=True, keep_sparse=False)
        for mat in ("KS", "S", "HCORE", "P")
        )
        res_dict = {
            "F": F, "P": P, "S": S, "H": H,
            "charge": 0,
            "atomic_numbers": elem_numbers,
            "geometry_bohr": coords_bohr
        }

        for m in ("F", "P", "S", "H"):
            if cutoff is not None:
                res_dict[m] = np.where(np.abs(res_dict[m]) <= cutoff, 0, res_dict[m])
            res_dict[m] = apply_perm_map(res_dict[m], perm_map)
    else:
        #norb_z = norb_by_z(method)
        #topk_blocks = get_topk_blocks(
        #    workdir, method, nao, atoms, T_vectors, norb_z, lattice_vecs_bohr, topk=topk, cutoff=cutoff
        #)
        T_matrices = {}
        for T_idx, T_vec in enumerate(T_vectors):
            F_T, P_T, S_T, H_T = (
                parse_matrix(workdir, method, key, nao, binary=True, T_idx=T_idx, keep_sparse=True)
                for key in ("KS", "P", "S", "HCORE")
            )
            T_matrices[tuple(T_vec)] = {"F": F_T, "P": P_T, "S": S_T, "H": H_T}
        
        if sym == "XYZ":
            pbc = [True, True, True]
        elif sym == "XY":
            pbc = [True, True, False]
        
        res_dict = {
            "nao": nao,
            "matrices": T_matrices,
            "charge": 0,
            "net_spin": 0,
            "pbc": pbc,
            "atomic_numbers": elem_numbers,
            "geometry_bohr": coords_bohr,
            "cell_bohr": lattice_vecs_bohr,
        }

    if get_energy:
        res_dict["energy_cp2k_Ha"] = energy_Ha

    if bandgap_info is not None:
        bg_res = parse_molog(workdir)
        res_dict["bandgap_pbe"] = bandgap_info.get("bandgap_pbe")
        res_dict["bandgap_hse"] = bandgap_info.get("bandgap_hse")
        res_dict["bandgap_cp2k"] = bg_res.bandgap_eV
        #res_dict["bandgap_gw"] = bandgap_info.get("bandgap_gw")
        #res_dict["gap_type_pbe"] = bandgap_info.get("gap_type_pbe")
        #res_dict["gap_type_hse"] = bandgap_info.get("gap_type_hse")

    # Save
    out_path = workdir / f"{out_name}.pkl"
    tmp_path = out_path.with_suffix(".tmp.pkl")

    with open(tmp_path, 'wb') as f:
        pickle.dump(res_dict, f)

    tmp_path.replace(out_path)

    # Clean up temporary matrices directory
    temp_dir = workdir / TEMP_SAVE_DIR
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return str(out_path)

def get_topk_blocks(workdir, method, nao, atoms, T_vectors, norb_z, lattice_vecs,
    topk=32, cutoff=1e-32):
    """
    Extract top-k interaction blocks for each atom from periodic DFT matrices.
    """
    permute_block = _permute_block_pbe if method == "pbe" else _permute_block

    n_atoms = len(atoms)

    blocks_by_source = defaultdict(list)
    atom_ranges = build_atom_ranges(norb_z, atoms)

    atom_coords = np.array([a['coord_bohr'] for a in atoms], dtype=np.float64)
    atom_numbers = np.array([a['atomnum'] for a in atoms], dtype=np.int32)

    a1, a2, a3 = lattice_vecs

    # Store
    T_matrix_cache = {}

    # Step 1: Score all blocks and cache matrices
    for T_idx, T_vec in enumerate(T_vectors):
        #T_norm = np.linalg.norm(T_vec)

        # Get matrices for the translation
        F_T, P_T, S_T, H_T = (
            parse_matrix(workdir, method, key, nao, binary=True, T_idx=T_idx, keep_sparse=True)
            for key in ("KS", "P", "S", "HCORE")
        )

        # Store matrices
        T_matrix_cache[T_idx] = {'KS': F_T, 'P': P_T, 'S': S_T, 'HCORE': H_T}

        # Extract atom-atom blocks
        for src in range(1, n_atoms + 1):
            for ngb in range(1, n_atoms + 1):
                cell = tuple(T_vec)

                # Compute norms
                nF = compute_block_norm_squared_sparse(F_T, src, ngb, atom_ranges)
                nP = compute_block_norm_squared_sparse(P_T, src, ngb, atom_ranges)
                nS = compute_block_norm_squared_sparse(S_T, src, ngb, atom_ranges)
                nH = compute_block_norm_squared_sparse(H_T, src, ngb, atom_ranges)
                score = float(nF * nS * nH)

                if score < cutoff:
                    continue

                # Store block metadata
                blocks_by_source[src].append({
                    'ngb': ngb,
                    'cell': cell,
                    'T_idx': T_idx,
                    'score': score,
                    'Fscr': nF, 'Pscr': nP, 'Sscr': nS, 'Hscr': nH
                })

    # Phase 2: Top-k selection and permutation
    topk_blocks: List[Dict[str, Any]] = []
    self_ctr, pair_ctr = 0, 0

    # Process each source atom
    for src in range(1, n_atoms + 1):
        atom_blocks = blocks_by_source.get(src, [])

        # Separate self-blocks from neighbor blocks
        self_block = None
        neighbor_blocks = []

        for block in atom_blocks:
            if block['ngb'] == src and block['cell'] == (0, 0, 0):
                self_block = block
            else:
                neighbor_blocks.append(block)

        # Add self-block if it exists
        if self_block is not None:
            T_idx = self_block['T_idx']
            ngb = self_block['ngb']
            z_src = atom_numbers[src - 1]
            z_ngb = atom_numbers[ngb - 1]
            F_block = get_block(T_matrix_cache[T_idx]["KS"], src, ngb, atom_ranges)
            P_block = get_block(T_matrix_cache[T_idx]["P"], src, ngb, atom_ranges)
            S_block = get_block(T_matrix_cache[T_idx]["S"], src, ngb, atom_ranges)
            H_block = get_block(T_matrix_cache[T_idx]["HCORE"], src, ngb, atom_ranges)

            topk_blocks.append({
                "is_self": True,
                "ctr": self_ctr,
                "source": src,
                "neighbor": ngb,
                "cell": self_block['cell'],
                "matrix": {
                    'F': permute_block(F_block, z_src, z_ngb), 'P': permute_block(P_block, z_src, z_ngb), 'S': permute_block(S_block, z_src, z_ngb), 'H': permute_block(H_block, z_src, z_ngb),
                    'score': self_block['score'],
                    'Fscr': self_block['Fscr'],
                    'Pscr': self_block['Pscr'],
                    'Sscr': self_block['Sscr'],
                    'Hscr': self_block['Hscr']
                },
            })
            self_ctr += 1

        # Sort neighbor blocks by score
        neighbor_blocks.sort(key=lambda b: b['score'], reverse=True)

        # Apply top-k with tie-breaking
        if len(neighbor_blocks) <= topk:
            selected_blocks = neighbor_blocks
        else:
            #kth_score = neighbor_blocks[topk - 1]['score']
            selected_blocks = sorted(neighbor_blocks, key=lambda b: b['score'], reverse=True)[:topk]
            #rtol = 1e-12
            #selected_blocks = [
            #    b for b in neighbor_blocks
            #   if b['score'] > kth_score or abs(b['score'] - kth_score) <= rtol * max(1.0, abs(kth_score))
            #]

        # Add selected blocks to output
        for block in selected_blocks:
            ngb = block['ngb']
            Rvec = block['cell']
            T_idx = block['T_idx']

            F_block = get_block(T_matrix_cache[T_idx]["KS"], src, ngb, atom_ranges)
            P_block = get_block(T_matrix_cache[T_idx]["P"], src, ngb, atom_ranges)
            S_block = get_block(T_matrix_cache[T_idx]["S"], src, ngb, atom_ranges)
            H_block = get_block(T_matrix_cache[T_idx]["HCORE"], src, ngb, atom_ranges)

            z_src = atom_numbers[src - 1]
            z_ngb = atom_numbers[ngb - 1]

            perm_mat = {
                'H': permute_block(H_block, z_src, z_ngb),
                'S': permute_block(S_block, z_src, z_ngb),
                'F': permute_block(F_block, z_src, z_ngb),
                'P': permute_block(P_block, z_src, z_ngb),
                'score': block['score'],
                'Fscr': block['Fscr'],
                'Pscr': block['Pscr'],
                'Sscr': block['Sscr'],
                'Hscr': block['Hscr']
            }

            # Calculate distance
            ic1, ic2, ic3 = Rvec
            r_i = atom_coords[src - 1]
            r_j = atom_coords[ngb - 1]
            shift = ic1 * a1 + ic2 * a2 + ic3 * a3
            dvec = r_j - r_i + shift
            dist = float(np.linalg.norm(dvec))

            topk_blocks.append({
                "is_self": False,
                "ctr": pair_ctr,
                "source": src,
                "neighbor": ngb,
                "cell": Rvec,
                "matrix": perm_mat,
                "dist": dist,
            })
            pair_ctr += 1

    return topk_blocks
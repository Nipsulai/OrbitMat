import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Any
import json

from .config import XTB_PATH, PBE_PATH, SCAN_PATH, TZVP_PATH, METHODS
from .npz_io import write_periodic, write_molecular
from .parse_cp2k import (
    Atom, BandgapResult, TEMP_SAVE_DIR,
    get_elem_symb_atomnumber_dict,
    parse_cp2k_output, _detect_uks, parse_matrix, parse_molog,
)
from .perm_and_blocks import _permute_block, _permute_block_pbe, _permute_block_scan, _permute_block_tzvp, apply_perm_map, get_perm_map, build_atom_ranges, get_block, compute_block_norm_squared_sparse

def norb_by_z(method):
    """Return {atomic_number: n_orbitals} for the given method's basis set."""
    cfg = METHODS[method]
    if cfg.norb_json_path is None:
        raise ValueError(f"No norb_json_path configured for method '{method}'")
    with open(cfg.norb_json_path) as f:
        data = json.load(f)
    if cfg.has_dft:
        return {int(z): norb["total"] for z, norb in data.items()}
    else:
        return {int(z): norb for z, norb in data.items()}


def postproc_matrices(workdir, method, sym, rspace=True, get_energy=True, cutoff=1e-32, out_name="matrices", compress=True, topk=32, bandgap_info=None, energy_dft_ha=None):
    workdir = Path(workdir)
    nao, atoms, coords_bohr, energy_Ha, T_vectors, K_vectors, K_weights, lattice_vecs_bohr = parse_cp2k_output(workdir, method, rspace)

    elem_numbers = np.array([a["atomnum"] for a in atoms])

    mcfg = METHODS[method]
    uks = _detect_uks(workdir, mcfg.project_name, mcfg.periodic, rspace)

    if not mcfg.periodic:
        # Molecular case
        _pm = lambda mat, sp=1: parse_matrix(
            workdir, method, mat, nao, binary=True, keep_sparse=False, rspace=True, spin=sp
        )
        S = _pm("S")
        H = _pm("HCORE")
        if uks:
            F_a, F_b = _pm("KS", 1), _pm("KS", 2)
            P_a, P_b = _pm("P", 1),  _pm("P", 2)
            spin_mats = {"F_a": F_a, "F_b": F_b, "P_a": P_a, "P_b": P_b, "S": S, "H": H}
        else:
            spin_mats = {"F": _pm("KS"), "P": _pm("P"), "S": S, "H": H}

        res_dict = {
            **spin_mats,
            "unrestricted": uks,
            "charge": 0,
            "atomic_numbers": elem_numbers,
            "geometry_bohr": coords_bohr,
        }

        if cutoff is not None:
            for key in res_dict:
                if isinstance(res_dict[key], np.ndarray):
                    res_dict[key] = np.where(np.abs(res_dict[key]) <= cutoff, 0, res_dict[key])
    else:
        if sym == "3D":
            pbc = [True, True, True]
        elif sym == "2D":
            pbc = [True, True, False]

        def _pm_per(mat_type, TK_idx, sp=1):
            return parse_matrix(
                workdir, method, mat_type, nao,
                binary=True, TK_idx=TK_idx, keep_sparse=True, rspace=rspace, spin=sp,
            )

        matrices = {}
        TK_vectors = T_vectors if rspace else K_vectors
        for TK_idx, TK_vec in enumerate(TK_vectors[:200]):
            S_TK = _pm_per("S",     TK_idx)
            H_TK = _pm_per("HCORE", TK_idx)
            if uks:
                matrices[tuple(TK_vec)] = {
                    "F_a": _pm_per("KS", TK_idx, 1),
                    "F_b": _pm_per("KS", TK_idx, 2),
                    "P_a": _pm_per("P",  TK_idx, 1),
                    "P_b": _pm_per("P",  TK_idx, 2),
                    "S": S_TK, "H": H_TK,
                }
            else:
                matrices[tuple(TK_vec)] = {
                    "F": _pm_per("KS", TK_idx),
                    "P": _pm_per("P",  TK_idx),
                    "S": S_TK, "H": H_TK,
                }

        res_dict = {
            "nao": nao,
            "matrices": matrices,
            "rspace": rspace,
            "kpoints_2pi_bohr": K_vectors,
            "unrestricted": uks,
            "charge": 0,
            "net_spin": 1 if uks else 0,
            "pbc": pbc,
            "atomic_numbers": elem_numbers,
            "geometry_bohr": coords_bohr,
            "cell_bohr": lattice_vecs_bohr,
        }

    if get_energy:
        res_dict["energy_cp2k_Ha"] = energy_Ha

    if energy_dft_ha is not None:
        res_dict["energy_dft_Ha"] = float(energy_dft_ha)

    if bandgap_info is not None:
        eigen_path = workdir / f"{METHODS[method].project_name}-eigenvalues-1_0.MOLog"
        res_dict["bandgap_pbe"] = bandgap_info.get("bandgap_pbe")
        res_dict["bandgap_hse"] = bandgap_info.get("bandgap_hse")
        if eigen_path is not None:
            bg_res = parse_molog(eigen_path)
            res_dict["bandgap_grid"] = bg_res.bandgap_eV
            res_dict["vbm_grid"]     = bg_res.vbm_eV
            res_dict["cbm_grid"]     = bg_res.cbm_eV
        else:
            raise ValueError(f"Warning: MOLog not found, skipping bandgap_grid for {workdir.name}")
        #res_dict["bandgap_gw"] = bandgap_info.get("bandgap_gw")
        #res_dict["gap_type_pbe"] = bandgap_info.get("gap_type_hse")
    # Save
    out_path = workdir / f"{out_name}.npz"
    tmp_path = workdir / f"{out_name}.tmp.npz"

    if mcfg.periodic:
        write_periodic(tmp_path, res_dict)
    else:
        write_molecular(tmp_path, res_dict)

    tmp_path.rename(out_path)

    # Clean up temporary matrices directory
    temp_dir = workdir / TEMP_SAVE_DIR
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return str(out_path)


def mcfg_for(method: str):
    return METHODS[method]
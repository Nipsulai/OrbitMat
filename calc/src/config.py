import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── CP2K execution ────────────────────────────────────────────────────────────
# Local defaults; override via env vars for cluster use:
#   CP2K_SETUP  path to toolchain setup script
#   CP2K_BIN    path to cp2k binary
SOURCE = Path("/home/mnouman/cp2k/tools/toolchain/install/setup").absolute()
SSMP   = Path("/home/mnouman/cp2k/build/bin/cp2k.psmp").absolute()
OUTPUT = "out.cp2k"

# ── Calculation parameters ────────────────────────────────────────────────────
TIMEOUT          = 3600
KPOINTS_ACC      = 1500   # accuracy for k-grid generation
KPOINTS_DENSITY  = 5.0    # k-points density for 3D grids, per 1/Å
VACUUM_PADDING   = 3      # Å, padding per side for molecular boxes

# ── File paths ────────────────────────────────────────────────────────────────
ELEM_PATH           = Path("calc/src/input/elements.txt").absolute()

# DZ (PBE)
BASIS_PATH          = Path("calc/input/basis/BASIS_MOLOPT_DZ").absolute()
POTENTIAL_PATH      = Path("calc/input/potential/POTENTIAL_DZ").absolute()
ELEM_DATA           = Path("calc/src/input/elem_data.json").absolute()
PBE_PATH            = Path("calc/src/input/basis/basis_molopt_dz.json").absolute()

# TZVP (PBE)
BASIS_PATH_TZVP     = Path("calc/src/input/basis/BASIS_MOLOPT_TZVP").absolute()
POTENTIAL_PATH_TZVP = Path("calc/src/input/potential/POTENTIAL_TZVP").absolute()
ELEM_DATA_TZVP      = Path("calc/src/input/elem_data_tzvp.json").absolute()
TZVP_PATH           = Path("calc/src/input/basis/basis_molopt_tzvp.json").absolute()

# SCAN
BASIS_PATH_SCAN     = Path("calc/src/input/basis/BASIS_MOLOPT_SCAN_DZ").absolute()
POTENTIAL_PATH_SCAN = Path("calc/src/input/potential/POTENTIAL_DZ_SCAN").absolute()
ELEM_DATA_SCAN      = Path("calc/src/input/elem_data_scan.json").absolute()
SCAN_PATH           = Path("calc/src/input/basis/basis_molopt_dz_scan.json").absolute()
# xTB
XTB_PATH            = Path("calc/src/input/xtb_basis.json").absolute()

# ── Basis family strings ──────────────────────────────────────────────────────
BASIS_FAMILY        = "DZ-MOLOPT-PBE-GTH-q"
GTH_FAMILY          = "GTH-PBE-q"
BASIS_FAMILY_SCAN   = "DZVP-MOLOPT-SCAN-GTH-q"
GTH_FAMILY_SCAN     = "GTH-SCAN-q"
BASIS_FAMILY_TZVP   = "TZVP-MOLOPT-PBE-GTH-q"
GTH_FAMILY_TZVP     = "GTH-PBE-q"


# ── Method registry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MethodConfig:
    periodic: bool          # CIF input (True) vs XYZ molecule (False)
    has_dft: bool           # Uses GTH basis/pseudopotentials (DZ / SCAN / TZVP)
    supports_band: bool     # Can run band-structure calculation
    project_name: str       # CP2K PROJECT keyword (determines prefix of output .csr/.MOLog files)
    template: str           # Template file stem: calc/src/input/template/{template}.inp
    basis_family: str       = ""
    gth_family: str         = ""
    basis_path: Optional[Path] = None
    potential_path: Optional[Path] = None
    elem_data_path: Optional[Path] = None
    norb_json_path: Optional[Path] = None
    # Orbital permutation identifier used by postproc / hdf5_utils
    orbital_perm: str       = "xtb"   # "dz" | "scan" | "tzvp" | "xtb"


METHODS: dict[str, MethodConfig] = {
    "dz": MethodConfig(
        periodic=True, has_dft=True, supports_band=True,
        project_name="dz",   template="dz",
        basis_family=BASIS_FAMILY, gth_family=GTH_FAMILY,
        basis_path=BASIS_PATH, potential_path=POTENTIAL_PATH,
        elem_data_path=ELEM_DATA, norb_json_path=PBE_PATH,
        orbital_perm="dz",
    ),
    "scan": MethodConfig(
        periodic=True, has_dft=True, supports_band=True,
        project_name="scan", template="scan",
        basis_family=BASIS_FAMILY_SCAN, gth_family=GTH_FAMILY_SCAN,
        basis_path=BASIS_PATH_SCAN, potential_path=POTENTIAL_PATH_SCAN,
        elem_data_path=ELEM_DATA_SCAN, norb_json_path=SCAN_PATH,
        orbital_perm="scan",
    ),
    "tzvp": MethodConfig(
        periodic=True, has_dft=True, supports_band=True,
        project_name="tzvp", template="tzvp",
        basis_family=BASIS_FAMILY_TZVP, gth_family=GTH_FAMILY_TZVP,
        basis_path=BASIS_PATH_TZVP, potential_path=POTENTIAL_PATH_TZVP,
        elem_data_path=ELEM_DATA_TZVP, norb_json_path=TZVP_PATH,
        orbital_perm="tzvp",
    ),
    "xtb": MethodConfig(
        periodic=True, has_dft=False, supports_band=True,
        project_name="xtb",  template="xtb",
        norb_json_path=XTB_PATH, orbital_perm="xtb",
    ),
    "dz_mol": MethodConfig(
        periodic=False, has_dft=True, supports_band=False,
        project_name="dz_mol", template="dz_mol",
        basis_family=BASIS_FAMILY, gth_family=GTH_FAMILY,
        basis_path=BASIS_PATH, potential_path=POTENTIAL_PATH,
        elem_data_path=ELEM_DATA, norb_json_path=PBE_PATH,
        orbital_perm="dz",
    ),
    "xtb_mol": MethodConfig(
        periodic=False, has_dft=False, supports_band=False,
        project_name="xtb_mol", template="xtb_mol",
        norb_json_path=XTB_PATH, orbital_perm="xtb",
    ),
    # Two-step xTB: retries with CHECK_ATOMIC_CHARGES F then T.
    # Reuses xtb template; project_name="xtb" so postproc reads xtb-*.csr files.
    "charge_xtb": MethodConfig(
        periodic=True, has_dft=False, supports_band=False,
        project_name="xtb",  template="xtb",
        norb_json_path=XTB_PATH, orbital_perm="xtb",
    ),
}

# Flat list of valid method names (kept for places that still need it)
VALID_METHODS: list[str] = list(METHODS)

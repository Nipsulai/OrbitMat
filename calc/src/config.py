from pathlib import Path

# CP2K execution
SOURCE = Path("cp2k/tools/toolchain/install/setup").absolute()
SSMP = Path("cp2k/exe/local/cp2k.ssmp").absolute()
OUTPUT = "out.cp2k"

BASIS_FAMILY = "DZ-MOLOPT-PBE-GTH-q"
GTH_FAMILY = "GTH-PBE-q"

BASIS_FAMILY_SCAN = "DZVP-MOLOPT-SCAN-GTH-q"
GTH_FAMILY_SCAN = "GTH-SCAN-q"

BASIS_FAMILY_TZVP = "TZVP-MOLOPT-PBE-GTH-q"
GTH_FAMILY_TZVP = "GTH-PBE-q"

# Calculation parameters
TIMEOUT = 1400
KPOINTS_ACC = 1500
# accuracy for kgrid generation
KPOINTS_DENSITY = 6.0 # k-points density for 2D materials kgrids, per 1/Å
VACUUM_PADDING = 3 # Ang, padding molecule for box, per side

# File paths
BASIS_PATH = Path("calc/input/BASIS_MOLOPT_DZ").absolute()
POTENTIAL_PATH = Path("calc/input/POTENTIAL_DZ").absolute()
ELEM_DATA = Path("calc/src/input/elem_data.json").absolute()
ELEM_PATH = Path("calc/src/input/elements.txt").absolute()
XTB_PATH = Path("calc/src/input/xtb_basis.json").absolute()
PBE_PATH = Path("calc/src/input/basis_molopt_dz.json").absolute()

BASIS_PATH_SCAN = Path("calc/src/input/BASIS_MOLOPT_SCAN_DZ").absolute()
POTENTIAL_PATH_SCAN = Path("calc/src/input/POTENTIAL_DZ_SCAN").absolute()
ELEM_DATA_SCAN = Path("calc/src/input/elem_data_scan.json").absolute()
SCAN_PATH = Path("calc/src/input/basis_molopt_dz_scan.json").absolute()

BASIS_PATH_TZVP = Path("calc/src/input/BASIS_MOLOPT_TZVP").absolute()
POTENTIAL_PATH_TZVP = Path("calc/src/input/POTENTIAL_TZVP").absolute()
ELEM_DATA_TZVP = Path("calc/src/input/elem_data_tzvp.json").absolute()
TZVP_PATH = Path("calc/src/input/basis_molopt_tzvp.json").absolute()

# Valid methods
VALID_METHODS = ["pbe", "xtb", "xyz", "pbemol", "charge_xtb", "scan", "tzvp"]
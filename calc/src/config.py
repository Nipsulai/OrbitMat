from pathlib import Path

# CP2K execution
SOURCE = Path("cp2k/tools/toolchain/install/setup").absolute()
SSMP = Path("cp2k/exe/local/cp2k.ssmp").absolute()
OUTPUT = "out.cp2k"

BASIS_FAMILY = "DZ-MOLOPT-PBE-GTH-q"
GTH_FAMILY = "GTH-PBE-q"

# Calculation parameters
TIMEOUT = 1000
KPOINTS_ACC = 1000 # accuracy for kgrid generation
KPOINTS_DENSITY = 6.0 # k-points density for 2D materials kgrids, per 1/Å
VACUUM_PADDING = 30.0  # Ang, padding around molecule for box

# File paths
BASIS_PATH = Path("calc/input/BASIS_MOLOPT_DZ").absolute()
POTENTIAL_PATH = Path("calc/input/POTENTIAL_DZ").absolute()
ELEM_DATA = Path("calc/src/input/elem_data.json").absolute()
ELEM_PATH = Path("calc/src/input/elements.txt").absolute()
XTB_PATH = Path("calc/src/input/xtb_basis.json").absolute()
PBE_PATH = Path("calc/src/input/basis_molopt_dz.json").absolute()

# Valid methods
VALID_METHODS = ["pbe", "xtb", "xyz"]
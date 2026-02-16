import json
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Kpoints

#Saved dummy input generating pythons script

# -------------------- CONFIG --------------------
CIF = "/home/nikolai/OrbitMat/data/cifs/files/Ag1As1S1_604740.cif"
PP_JSON = "input/cutoff.json"    # your JSON with Zval + cutoff per element
BASIS_FAMILY = "DZ-MOLOPT-PBE-GTH-q"
GTH_FAMILY = "GTH-PBE-q"
OUT = "auto_input.inp"
# -------------------------------------------------

def main():
    # ----------------- Load structure -----------------
    struct = Structure.from_file(CIF)
    # ----------------- Elements -----------------
    elements = [str(site.specie) for site in struct.sites]
    unique = sorted(set(elements))
    print("Elements in primitive cell:", unique)

    # ----------------- Load pseudopotential metadata -----------------
    pp_data = json.load(open(PP_JSON))

    # Validate elements
    for elem in unique:
        if elem not in pp_data:
            raise RuntimeError(f"Element {elem} not found in {PP_JSON}")

    # q-values come from Zval field in JSON
    elem_q = {elem: pp_data[elem]["Zval"] for elem in unique}

    # Total electrons and UKS flag
    total_e = sum(pp_data[elem]["Zval"] * elements.count(elem) for elem in unique)
    UKS = "TRUE" if (total_e % 2 == 1) else "FALSE"
    print("Total valence electrons =", total_e, "| UKS =", UKS)

    # ----------------- Automatic KPOINTS -----------------
    kpts = Kpoints.automatic_density(struct, kppa=1000)
    kx, ky, kz = kpts.kpts[0]
    print(f"Automatic KPOINTS grid: {kx} {ky} {kz}")

    # ----------------- Plane-wave cutoff -----------------
    max_cutoff = max(pp_data[elem]["cutoff"] for elem in unique)
    print("Max cutoff among elements:", max_cutoff)

    # ------------------ Write CP2K input file -------------------
    with open(OUT, "w") as f:

        # ---------- GLOBAL & DFT ----------
        f.write(f"""&GLOBAL
  PROJECT CIF_INPUT
  RUN_TYPE ENERGY
  PRINT_LEVEL LOW
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep

  &DFT
    BASIS_SET_FILE_NAME home/nikolai/OrbitMat/data/input/BASIS_MOLOPT_DZ
    POTENTIAL_FILE_NAME home/nikolai/OrbitMat/data/input/POTENTIAL_DZ
    UKS {UKS}

    &MGRID
      CUTOFF {max_cutoff}
      REL_CUTOFF 60
    &END MGRID

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC

    &KPOINTS
      SCHEME MONKHORST-PACK {kx} {ky} {kz}
      FULL_GRID T
    &END KPOINTS

    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1e-8
      MAX_SCF 200

      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION

      &MIXING
        METHOD BROYDEN_MIXING
        ALPHA 0.05
        NBROYDEN 8
      &END MIXING

      ADDED_MOS 20
    &END SCF

    &PRINT
      &MO
        EIGENVALUES T
        OCCNUMS T
        FILENAME eigenvalues
        ADD_LAST NUMERIC
        &EACH
          QS_SCF 0
        &END EACH
      &END MO
    &END PRINT

  &END DFT

  &SUBSYS
""")

        # ---------- CELL from CIF ----------
        f.write(f"""
    &CELL
      PERIODIC XYZ
      CELL_FILE_FORMAT CIF
      CELL_FILE_NAME {CIF}
    &END CELL
""")

        # ---------- COORD from CIF ----------
        f.write(f"""
    &TOPOLOGY
      COORD_FILE_FORMAT CIF
      COORD_FILE_NAME {CIF}
      &GENERATE
        REORDER T
      &END GENERATE
    &END TOPOLOGY
""")

        # ---------- KIND blocks ----------
        for elem in unique:
            q = elem_q[elem]
            f.write(f"""
    &KIND {elem}
      BASIS_SET {BASIS_FAMILY}{q}
      POTENTIAL {GTH_FAMILY}{q}
    &END KIND
""")

        f.write("""
  &END SUBSYS
&END FORCE_EVAL
""")

    print("Written:", OUT)
    print("Element → q:", elem_q)
    print("CUTOFF used =", max_cutoff)
    print("KPOINTS =", (kx, ky, kz))


if __name__ == "__main__":
    main()
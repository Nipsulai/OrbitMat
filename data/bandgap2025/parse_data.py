"""
Parse hse06_2.db + pbesol_2.db → CIF files + all_data.json.

Equivalent to data/scripts/parse_data.py but adapted for ASE .db sources.
No k-point/calculator metadata is stored in these DBs (calculator=unknown).

Flags:
  --reduce   Write the primitive standard cell CIF for reducible structures
             instead of the conventional cell. Sanity-checks by reloading
             the written CIF and comparing atom count.
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ase.db import connect
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HSE_DB   = Path(__file__).parent / "hse06_bandgap.db"
PBE_DB   = Path(__file__).parent / "pbesol_bandgap.db"
OUT_DIR  = Path(__file__).parent / "cifs" / "all_files"
JSON_OUT = Path(__file__).parent / "all_data.json"

SYMPREC   = 0.1   # same as Materials Project default
ANGLE_TOL = 5.0

SG_TO_BRAVAIS = {
    # Triclinic
    1: ("simple", "triclinic"), 2: ("simple", "triclinic"),
    # Monoclinic
    3: ("simple", "monoclinic"), 4: ("simple", "monoclinic"),
    5: ("base",   "monoclinic"), 6: ("simple", "monoclinic"),
    7: ("simple", "monoclinic"), 8: ("base",   "monoclinic"),
    9: ("base",   "monoclinic"), 10: ("simple", "monoclinic"),
    11: ("simple","monoclinic"), 12: ("base",   "monoclinic"),
    13: ("simple","monoclinic"), 14: ("simple", "monoclinic"),
    15: ("base",  "monoclinic"),
    # Orthorhombic
    **{sg: ("simple", "orthorhombic") for sg in [
        16,17,18,19,25,26,27,28,29,30,31,32,33,34,
        47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]},
    **{sg: ("base",   "orthorhombic") for sg in [20,21,35,36,37,38,39,40,41,63,64,65,66,67,68]},
    **{sg: ("face",   "orthorhombic") for sg in [22,42,43,69,70]},
    **{sg: ("body",   "orthorhombic") for sg in [23,24,44,45,46,71,72,73,74]},
    # Tetragonal
    **{sg: ("simple", "tetragonal") for sg in [
        75,76,77,78,81,83,84,85,86,89,90,91,92,93,94,95,96,
        99,100,101,102,103,104,105,106,111,112,113,114,115,116,117,118,
        123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138]},
    **{sg: ("body",   "tetragonal") for sg in [
        79,80,82,87,88,97,98,107,108,109,110,119,120,121,122,139,140,141,142]},
    # Trigonal / Hexagonal
    **{sg: ("0", "rhombohedral") for sg in [146,148,155,160,161,166,167]},
    **{sg: ("0", "hexagonal")    for sg in list(range(143,146)) + list(range(147,148)) +
                                            list(range(149,160)) + list(range(162,166)) +
                                            list(range(168,195))},
    # Cubic
    **{sg: ("simple", "cubic") for sg in [195,198,200,201,205,207,208,212,213,215,218,221,222,223,224]},
    **{sg: ("face",   "cubic") for sg in [196,202,203,209,210,216,219,225,226,227,228]},
    **{sg: ("body",   "cubic") for sg in [197,199,204,206,211,214,217,220,229,230]},
}


def space_group_to_bravais(sg_num):
    return SG_TO_BRAVAIS.get(sg_num, ("unknown", "unknown"))


def get_symmetry(struct):
    sga = SpacegroupAnalyzer(struct, symprec=SYMPREC, angle_tolerance=ANGLE_TOL)
    sg_num    = sga.get_space_group_number()
    sg_symbol = sga.get_space_group_symbol()
    crystal   = sga.get_crystal_system()
    prim      = sga.get_primitive_standard_structure()
    reduced   = len(prim) < len(struct)
    multiplicity = len(struct) / len(prim)
    return sg_num, sg_symbol, crystal, prim, reduced, multiplicity


def write_cif_verified(struct, cif_path):
    """Write a CIF and reload it to verify atom count matches. Returns (path, ok, msg)."""
    CifWriter(struct).write_file(str(cif_path))
    reloaded = Structure.from_file(str(cif_path))
    if len(reloaded) != len(struct):
        return cif_path, False, (
            f"Reload mismatch: wrote {len(struct)} atoms, reloaded {len(reloaded)}"
        )
    return cif_path, True, None


def row_to_record(hse_row, pbe_row, cif_path, struct):
    """Build a metadata dict from a paired HSE + PBEsol row."""
    lat = struct.lattice

    sg_num, sg_symbol, crystal, prim, reduced, multiplicity = get_symmetry(struct)
    center, lattice_sys = space_group_to_bravais(sg_num)

    # --- scalar fields available in the ASE DB ---
    def safe(row, key):
        return row.key_value_pairs.get(key)

    magmom = getattr(hse_row, "magmom", None)
    uks = bool(magmom is not None and magmom != 0.0)

    return {
        # identity
        "db_id_hse":    hse_row.id,
        "db_id_pbesol": pbe_row.id,
        "unique_id_hse":    hse_row.unique_id,
        "unique_id_pbesol": pbe_row.unique_id,
        "red_formula":  safe(hse_row, "red_formula"),
        "formula":      struct.formula,
        "nspecies":     safe(hse_row, "nspecies"),
        "cif_path":     str(cif_path),

        # band gap + edges + energetics (eV)
        "bandgap_info": {
            "bandgap_hse":    safe(hse_row, "band_gap"),
            "bandgap_pbesol": safe(pbe_row, "band_gap"),
            "cbm_hse":        safe(hse_row, "cbm"),
            "vbm_hse":        safe(hse_row, "vbm"),
            "e_fermi_hse":    safe(hse_row, "e_fermi"),
            "cbm_pbesol":     safe(pbe_row, "cbm"),
            "vbm_pbesol":     safe(pbe_row, "vbm"),
            "e_fermi_pbesol": safe(pbe_row, "e_fermi"),
            "energy_hse":     hse_row.energy,
            "energy_pbesol":  pbe_row.energy,
            "volume":         struct.volume,
            "density":        struct.density,
        },

        # DOS shape descriptors
        "skew_dos_hse":  safe(hse_row, "skew_dos"),
        "kurt_dos_hse":  safe(hse_row, "kurt_dos"),
        "skew_dos_pbesol": safe(pbe_row, "skew_dos"),
        "kurt_dos_pbesol": safe(pbe_row, "kurt_dos"),

        # charge / bonding descriptors
        "std_charge_hse":    safe(hse_row, "std_charge"),
        "min_dist_hse":      safe(hse_row, "min_dist"),
        "std_charge_pbesol": safe(pbe_row, "std_charge"),
        "min_dist_pbesol":   safe(pbe_row, "min_dist"),

        # structure (from HSE geometry, same for both)
        "natoms": len(struct),
        "lattice": {
            "a": lat.a, "b": lat.b, "c": lat.c,
            "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma,
            "matrix": lat.matrix.tolist(),
        },
        "lat_a_db": safe(hse_row, "lat_a"),
        "lat_b_db": safe(hse_row, "lat_b"),
        "lat_c_db": safe(hse_row, "lat_c"),

        # symmetry
        "space_group_num":    sg_num,
        "space_group_symbol": sg_symbol,
        "crystal_system":     crystal,
        "bravais": {"center": center, "lattice": lattice_sys, "space_group": sg_num},

        # primitive reduction
        "is_reducible":  reduced,
        "multiplicity":  multiplicity,
        "natoms_prim":   len(prim),

        # spin
        "magmom": magmom,
        "uks": uks,

        # calculator info (none available in these DBs)
        "calculator": hse_row.calculator or "unknown",
        "calculator_parameters": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Parse bandgap2025 ASE DBs to CIFs + JSON")
    parser.add_argument(
        "--reduce", action="store_true",
        help="For reducible structures write the primitive standard cell CIF "
             "instead of the conventional cell (verified by reloading).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hse_db = connect(str(HSE_DB))
    pbe_db = connect(str(PBE_DB))

    hse_rows = list(hse_db.select())
    pbe_rows = list(pbe_db.select())

    print(f"Loaded {len(hse_rows)} HSE and {len(pbe_rows)} PBEsol structures")

    # Match by red_formula (unique in both DBs, but row order differs)
    pbe_by_formula = {r.key_value_pairs.get("red_formula"): r for r in pbe_rows}
    unmatched_pbe = sum(
        1 for h in hse_rows
        if h.key_value_pairs.get("red_formula") not in pbe_by_formula
    )
    if unmatched_pbe:
        print(f"Warning: {unmatched_pbe} HSE rows have no PBEsol match")

    if args.reduce:
        print("--reduce enabled: primitive cells will be written for reducible structures")

    adaptor = AseAtomsAdaptor()
    records = []
    n_reduced = 0
    n_reload_fail = 0

    for hse_row in tqdm(hse_rows, desc="Processing"):
        formula = hse_row.key_value_pairs.get("red_formula")
        pbe_row = pbe_by_formula.get(formula)
        if pbe_row is None:
            print(f"Warning: no PBEsol match for id={hse_row.id} ({formula}), skipping")
            continue
        atoms = hse_row.toatoms()
        try:
            struct = adaptor.get_structure(atoms)
        except Exception as e:
            print(f"Warning: skipping id={hse_row.id} ({hse_row.key_value_pairs.get('red_formula')}): {e}")
            continue

        cif_name = f"{hse_row.id}_{hse_row.key_value_pairs.get('red_formula', 'unknown')}.cif"
        cif_path = OUT_DIR / cif_name

        try:
            record = row_to_record(hse_row, pbe_row, cif_path, struct)
        except Exception as e:
            print(f"Warning: metadata failed for id={hse_row.id}: {e}")
            continue

        # Choose which structure to write to CIF
        if args.reduce and record["is_reducible"]:
            sga = SpacegroupAnalyzer(struct, symprec=SYMPREC, angle_tolerance=ANGLE_TOL)
            struct_to_write = sga.get_primitive_standard_structure()
            n_reduced += 1
        else:
            struct_to_write = struct

        try:
            _, ok, msg = write_cif_verified(struct_to_write, cif_path)
            if not ok:
                print(f"Warning: reload check failed for id={hse_row.id}: {msg}")
                n_reload_fail += 1
        except Exception as e:
            print(f"Warning: CIF write failed for id={hse_row.id}: {e}")
            cif_path = None
            record["cif_path"] = None

        records.append(record)

    with open(JSON_OUT, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Done. {len(records)} records saved to {JSON_OUT}")
    print(f"CIF files in {OUT_DIR}")
    if args.reduce:
        print(f"  Reduced to primitive: {n_reduced} structures")
    if n_reload_fail:
        print(f"  Reload mismatches: {n_reload_fail}")


if __name__ == "__main__":
    main()

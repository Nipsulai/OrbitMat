"""
Parse a C2DB ASE database → CIF files + all_data.json.

Follows the same JSON schema as data/bandgap2025/parse_data.py.
Symmetry info (space group, bravais) is read directly from the DB.
Pymatgen is used only for CIF writing and primitive reduction.

Usage:
    python parse_data.py
    python parse_data.py --db /path/to/c2db.db --reduce
    python parse_data.py --db /path/to/c2db.db --selection "gap_hse>0"
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
DB_PATH  = Path(__file__).parent / "c2db.db"
OUT_DIR  = Path(__file__).parent / "cifs"
JSON_OUT = Path(__file__).parent / "all_data.json"

SYMPREC   = 0.1
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


def write_cif_verified(struct, cif_path):
    """Write CIF and reload to verify atom count. Returns (path, ok, msg)."""
    CifWriter(struct).write_file(str(cif_path))
    reloaded = Structure.from_file(str(cif_path))
    if len(reloaded) != len(struct):
        return cif_path, False, (
            f"Reload mismatch: wrote {len(struct)} atoms, reloaded {len(reloaded)}"
        )
    return cif_path, True, None


def row_to_record(row, cif_path, struct, prim, reduced, multiplicity):
    """Build a metadata dict from a C2DB row."""
    lat = struct.lattice

    def safe(key):
        v = row.get(key)
        return float(v) if v is not None else None

    # Symmetry from DB directly
    sg_num    = row.get("number")
    sg_symbol = row.get("international")
    center, lattice_sys = SG_TO_BRAVAIS.get(sg_num, ("unknown", "unknown")) if sg_num else ("unknown", "unknown")

    # UKS: use is_magnetic flag from DB
    is_magnetic = row.get("is_magnetic", False)
    magmom      = safe("magmom")
    uks = bool(is_magnetic)

    uid = row.get("uid") or f"id_{row.id}"

    return {
        # identity
        "db_id":    row.id,
        "unique_id": uid,
        "formula":  struct.formula,
        "cif_path": str(cif_path),
        "label":    row.get("label"),

        # band gaps, band edges, fermi levels (eV)
        "bandgap_info": {
            "bandgap_pbe":     safe("gap"),
            "bandgap_hse":     safe("gap_hse"),
            "bandgap_pbe_dir": safe("gap_dir"),
            "bandgap_hse_dir": safe("gap_dir_hse"),
            "bandgap_nosoc":   safe("gap_dir_nosoc"),
            "vbm_pbe":         safe("vbm"),
            "cbm_pbe":         safe("cbm"),
            "vbm_hse":         safe("vbm_hse"),
            "cbm_hse":         safe("cbm_hse"),
            "efermi_pbe":      safe("efermi"),
            "efermi_hse":      safe("efermi_hse"),
            "energy":          row.energy if hasattr(row, "energy") else None,
            "hform":           safe("hform"),
            "ehull":           safe("ehull"),
            "volume":          struct.volume,
            "density":         struct.density,
        },

        # structure
        "natoms": len(struct),
        "lattice": {
            "a": lat.a, "b": lat.b, "c": lat.c,
            "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma,
            "matrix": lat.matrix.tolist(),
        },

        # 3D symmetry (from DB)
        "space_group_num":    sg_num,
        "space_group_symbol": sg_symbol,
        "bravais": {"center": center, "lattice": lattice_sys, "space_group": sg_num},

        # 2D symmetry
        "layergroup":    row.get("layergroup"),
        "lgnum":         row.get("lgnum"),
        "bravais_2d":    row.get("bravais_type"),

        # primitive reduction
        "is_reducible":  reduced,
        "multiplicity":  multiplicity,
        "natoms_prim":   len(prim),

        # stability
        "dyn_stab": row.get("dyn_stab"),

        # spin
        "magmom": magmom,
        "is_magnetic": bool(is_magnetic),
        "uks":    uks,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse C2DB ASE database to CIFs + all_data.json")
    parser.add_argument("--db",      type=Path, default=DB_PATH,
                        help="Path to C2DB .db file")
    parser.add_argument("--out",     type=Path, default=JSON_OUT,
                        help="Output all_data.json path")
    parser.add_argument("--cif_dir", type=Path, default=OUT_DIR,
                        help="Output directory for CIF files")
    parser.add_argument("--selection", default="",
                        help="ASE DB selection string, e.g. 'gap_hse>0'")
    parser.add_argument("--reduce", action="store_true",
                        help="Write primitive standard cell CIF for reducible structures")
    args = parser.parse_args()

    args.cif_dir.mkdir(parents=True, exist_ok=True)

    db   = connect(str(args.db))
    rows = list(db.select(args.selection) if args.selection else db.select())
    print(f"Loaded {len(rows)} rows from {args.db.name}")

    if args.reduce:
        print("--reduce enabled: primitive cells written for reducible structures")

    adaptor = AseAtomsAdaptor()
    records = []
    n_reduced = n_reload_fail = n_errors = 0

    for row in tqdm(rows, desc="Processing"):
        uid = row.get("uid") or f"id_{row.id}"
        try:
            struct = adaptor.get_structure(row.toatoms())
        except Exception as e:
            print(f"Warning: skipping {uid}: {e}")
            n_errors += 1
            continue

        # Primitive reduction (for --reduce flag and multiplicity)
        try:
            sga  = SpacegroupAnalyzer(struct, symprec=SYMPREC, angle_tolerance=ANGLE_TOL)
            prim = sga.get_primitive_standard_structure()
            reduced      = len(prim) < len(struct)
            multiplicity = len(struct) / len(prim)
        except Exception:
            prim, reduced, multiplicity = struct, False, 1.0

        cif_path = args.cif_dir / f"{uid}.cif"

        try:
            record = row_to_record(row, cif_path, struct, prim, reduced, multiplicity)
        except Exception as e:
            print(f"Warning: metadata failed for {uid}: {e}")
            n_errors += 1
            continue

        struct_to_write = prim if (args.reduce and reduced) else struct
        if args.reduce and reduced:
            n_reduced += 1

        try:
            _, ok, msg = write_cif_verified(struct_to_write, cif_path)
            if not ok:
                print(f"Warning: reload check failed for {uid}: {msg}")
                n_reload_fail += 1
        except Exception as e:
            print(f"Warning: CIF write failed for {uid}: {e}")
            record["cif_path"] = None

        records.append(record)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone. {len(records)} records → {args.out}")
    print(f"CIF files → {args.cif_dir}")
    if args.reduce:
        print(f"  Reduced to primitive: {n_reduced}")
    if n_reload_fail:
        print(f"  Reload mismatches:    {n_reload_fail}")
    if n_errors:
        print(f"  Errors skipped:       {n_errors}")


if __name__ == "__main__":
    main()

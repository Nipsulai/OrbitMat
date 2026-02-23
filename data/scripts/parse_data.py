import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

from pymatgen.io.cif import CifWriter
from tqdm import tqdm

from helpers import (
    parse_json_structure,
    reduce_to_primitive,
    build_metadata_entry,
)

def main():
    parser = argparse.ArgumentParser(description="Band Gap Parser")
    parser.add_argument("--dir", default="data/bandgap")
    args = parser.parse_args()

    OUTPUT_DIR = Path("data/cifs").absolute()
    PRIM_DIR = OUTPUT_DIR / "all_files"
    PRIM_DIR.mkdir(parents=True, exist_ok=True)

    # To check whether a structure is primitive
    SYMPREC = 0.1 # Same as Materials Project, default: 0.01
    ANGLE_TOL = 5 # Default

    # Parse JSON files
    json_files = sorted(Path(args.dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    results = []
    structures = {}

    for json_path in tqdm(json_files, desc="Parsing JSON"):
        row = parse_json_structure(json_path)
        struct = row.pop("structure")
        structures[row["filename"]] = struct
        results.append(row)

    df = pd.DataFrame(results)

    # Write CIF + metadata
    meta_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        fname = row["filename"]
        struct = structures[fname]

        # Check if reducable to primitive
        prim, if_reducable = reduce_to_primitive(struct, SYMPREC, ANGLE_TOL)
        multiplicity = len(struct) / len(prim)

        # Write structure to CIF
        cif_path = PRIM_DIR / fname.replace(".json", ".cif")
        CifWriter(struct).write_file(cif_path)
 
        # Sanity check
        #reloaded = Structure.from_file(cif_path)
        #if len(reloaded) != len(struct):
        #    raise RuntimeError(
        #        f"Reload mismatch in {fname}: original={len(struct)} vs reloaded={len(reloaded)}"
        #    )
        meta = build_metadata_entry(
            row=row,
            struct_json=struct,
            cif_path=cif_path,
            if_reducable=if_reducable,
            multiplicity=multiplicity,
        )
        meta_records.append(meta)

    # Save data
    with open(OUTPUT_DIR / "all_data.json", "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"Done. Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
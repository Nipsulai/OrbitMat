"""CP2K input file generator for PBE, xTB (periodic), and xyz (molecular) methods."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.ase import AseAtomsAdaptor


from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.io import read
from io import StringIO

from .config import (
    ELEM_DATA, BASIS_FAMILY, GTH_FAMILY, KPOINTS_ACC, KPOINTS_DENSITY,
    BASIS_PATH, POTENTIAL_PATH, VACUUM_PADDING, VALID_METHODS
)

class CP2KInputGenerator:
    def __init__(
        self,
        method: str,
        sym: str,
        template_file: str = None,
        logger: logging.Logger = None,
    ):
        self.method = method.lower()
        self.sym = sym
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method: {method}. Must be one of {VALID_METHODS}")

        self.logger = logger or logging.getLogger(__name__)
        self.elem_data = None

        if self.method == "pbe":
            with open(ELEM_DATA, "r") as f:
                self.elem_data = json.load(f)

        with open(template_file, "r") as f:
            self.template = f.read()

    def load_structure(self, cif_path: str) -> Tuple[Structure, List[str], Dict[str, int]]:
        if not Path(cif_path).exists():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

        struct = Structure.from_file(cif_path)
        elements = [str(site.specie) for site in struct.sites]
        unique = sorted(set(elements))
        counts = {elem: elements.count(elem) for elem in unique}
        return struct, unique, counts

    def load_molecule(self, xyz_path: str) -> Tuple[Molecule, List[str], Dict[str, int]]:
        if not Path(xyz_path).exists():
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

        mol = Molecule.from_file(xyz_path)
        elements = [str(site.specie) for site in mol.sites]
        unique = sorted(set(elements))
        counts = {elem: elements.count(elem) for elem in unique}
        return mol, unique, counts

    def compute_box_size(
        self, mol: Molecule, padding: float = VACUUM_PADDING
    ) -> Tuple[int, int, int]:
        """
        Compute a CUBIC simulation box size for a molecule.
        Finds the largest dimension and applies it to all sides.
        """
        coords = mol.cart_coords
        spans = coords.max(axis=0) - coords.min(axis=0)
        
        max_span = np.max(spans)
        side_length = max_span + 2 * padding
        
        min_box_val = 2 * padding
        side_length = max(side_length, min_box_val)
        
        cubic_side = int(np.ceil(side_length))
        
        return cubic_side, cubic_side, cubic_side

    def compute_electronic_config(
        self, unique_elements: List[str], counts: Dict[str, int]
    ) -> Tuple[int, bool]:
        if self.method != "pbe":
            return 0, False, None
        missing = {e for e in unique_elements if e not in self.elem_data}
        if missing:
            return 0, False, missing
        total_e = sum(self.elem_data[e]["Zval"] * counts[e] for e in unique_elements)
        return total_e, (total_e % 2 == 1), None

    def _to_odd(self, n: int) -> int:
        return n + 1 - (n % 2)

    def compute_kpoints(self, struct: Structure, sym, bravais):
        if sym == "XYZ":
            kpts = Kpoints.automatic_density(struct, kppa=KPOINTS_ACC)
            kx, ky, kz = kpts.kpts[0]
            if bravais["center"] == "face" or bravais["lattice"] in ["rhombohedral", "hexagonal"]:
                kx, ky, kz = self._to_odd(int(kx)), self._to_odd(int(ky)), self._to_odd(int(kz)) 
        elif sym == "XY":
            atoms = AseAtomsAdaptor.get_atoms(struct)
            kpts = kptdensity2monkhorstpack(atoms, kptdensity=KPOINTS_DENSITY)
            kx, ky, kz = self._to_odd(int(kpts[0])), self._to_odd(int(kpts[1])), 1
        return kx, ky, kz

    def get_max_cutoff(self, unique_elements: List[str]) -> float:
        return max(self.elem_data[e]["cutoff"] for e in unique_elements)

    def get_element_charges(self, unique_elements: List[str]) -> Dict[str, int]:
        return {e: self.elem_data[e]["Zval"] for e in unique_elements}

    def write_input_pbe(
        self,
        output_path: str,
        cif_path: str,
        max_cutoff: float,
        kx: int, ky: int, kz: int,
        elem_q: Dict[str, int],
        unique_elements: List[str],
        unk_spins: bool,
        folder: str,
    ) -> None:
        kinds_str = ""
        for elem in unique_elements:
            q = elem_q[elem]
            kinds_str += f"""
    &KIND {elem}
      BASIS_SET {BASIS_FAMILY}{q}
      POTENTIAL {GTH_FAMILY}{q}
    &END KIND
"""
        output_content = self.template.format(
            BASIS_SET_FILE_NAME=BASIS_PATH,
            POTENTIAL_FILE_NAME=POTENTIAL_PATH,
            UKS="TRUE" if unk_spins else "FALSE",
            CUTOFF=int(max_cutoff),
            KX=kx, KY=ky, KZ=kz,
            CIF_PATH=cif_path,
            KINDS=kinds_str,
            FOLDER=folder,
        )
        with open(output_path, "w") as f:
            f.write(output_content)

    def write_input_xtb(
        self,
        output_path: str,
        cif_path: str,
        kx: int, ky: int, kz: int,
        folder: str,
        periodic: str
    ) -> None:
        output_content = self.template.format(KX=kx, KY=ky, KZ=kz, CIF_PATH=cif_path, FOLDER=folder, PERIODIC=periodic)
        with open(output_path, "w") as f:
            f.write(output_content)

    def write_input_xyz(
        self,
        output_path: str,
        xyz_path: str,
        ax: float, ay: float, az: float,
        folder: str
    ) -> None:
        output_content = self.template.format(
            XYZ_PATH=xyz_path,
            AX=f"{ax:.6f}",
            AY=f"{ay:.6f}",
            AZ=f"{az:.6f}",
            FOLDER=folder
        )
        with open(output_path, "w") as f:
            f.write(output_content)

    def generate_input(
        self, input_path: str, output_path: str, bravais, skip_if_unk: bool = False
    ) -> Dict:
        """Generate CP2K input file from structure/molecule."""
        metadata = {
            "skipped": False,
        }
        # Use relative path to avoid CP2K path length limits
        folder_abs = Path(output_path).resolve().parent / "matrices"
        folder_abs.mkdir(exist_ok=True)
        folder_rel = "matrices"

        if self.method == "xyz":
            mol, unique, counts = self.load_molecule(input_path)

            # xtb does not use box size so this is purely cosmetic
            ax, ay, az = self.compute_box_size(mol)

            metadata["elements"] = counts
            metadata["n_atoms"] = len(mol)
            metadata["box_size"] = {"x": ax, "y": ay, "z": az}

            self.write_input_xyz(output_path, input_path, ax, ay, az, folder_rel)
            return metadata

        struct, unique, counts = self.load_structure(input_path)
        
        kx, ky, kz = self.compute_kpoints(struct, self.sym, bravais) 
        metadata["kpoints"] = {"x": kx, "y": ky, "z": kz}
        metadata["elements"] = counts

        if self.method == "pbe":
            total_e, unk_spins, missing = self.compute_electronic_config(unique, counts)

            if missing:
                metadata["skipped"] = True
                metadata["skip_reason"] = f"Missing elems: {', '.join(sorted(missing))}"
                return metadata

            metadata["total_valence_electrons"] = total_e
            metadata["unrestricted"] = unk_spins

            if skip_if_unk and unk_spins:
                metadata["skipped"] = True
                metadata["skip_reason"] = "UKS required"
                return metadata

            max_cutoff = self.get_max_cutoff(unique)
            elem_q = self.get_element_charges(unique)
            metadata["cutoff"] = max_cutoff
            metadata["element_charges"] = elem_q

            self.write_input_pbe(
                output_path, input_path, max_cutoff,
                kx, ky, kz, elem_q, unique, unk_spins, folder_rel
            )
        else:  # xtb
            self.write_input_xtb(output_path, input_path, kx, ky, kz, folder_rel, periodic=self.sym)
        return metadata
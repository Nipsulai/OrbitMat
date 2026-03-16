"""CP2K input file generator for PBE, xTB (periodic), and xyz (molecular) methods."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import math

from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.ase import AseAtomsAdaptor


from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.io import read
from io import StringIO

from .config import (
    ELEM_DATA, BASIS_FAMILY, GTH_FAMILY, KPOINTS_ACC, KPOINTS_DENSITY,
    BASIS_PATH, POTENTIAL_PATH, VACUUM_PADDING, VALID_METHODS,
    ELEM_DATA_SCAN, BASIS_FAMILY_SCAN, GTH_FAMILY_SCAN,
    BASIS_PATH_SCAN, POTENTIAL_PATH_SCAN,
    ELEM_DATA_TZVP, BASIS_FAMILY_TZVP, GTH_FAMILY_TZVP,
    BASIS_PATH_TZVP, POTENTIAL_PATH_TZVP,
)

class CP2KInputGenerator:
    def __init__(
        self,
        method: str,
        sym: str,
        rspace: bool = True,
        template_file: str = None,
        logger: logging.Logger = None,
    ):
        self.method = method.lower()
        self.sym = sym
        self.rspace = rspace
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method: {method}. Must be one of {VALID_METHODS}")

        self.logger = logger or logging.getLogger(__name__)
        self.elem_data = None

        if self.method in {"pbe", "pbemol"}:
            with open(ELEM_DATA, "r") as f:
                self.elem_data = json.load(f)
        elif self.method == "scan":
            with open(ELEM_DATA_SCAN, "r") as f:
                self.elem_data = json.load(f)
        elif self.method == "tzvp":
            with open(ELEM_DATA_TZVP, "r") as f:
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
        self, mol, padding: float = 5.0
    ) -> Tuple[int, int, int]:
        """
        Compute an ORTHORHOMBIC simulation box size for a molecule.
        Calculates independent lengths for X, Y, and Z.
        """
        coords = mol.cart_coords
        # Calculate the span (max - min) for each axis independently
        spans = coords.max(axis=0) - coords.min(axis=0)
        
        # Apply padding to both sides of each axis
        # Resulting lengths: [L_x, L_y, L_z]
        side_lengths = spans + (2 * padding)
        
        # Optimization: CP2K's FFT solvers prefer even integers or 
        # numbers with small prime factors (2, 3, 5).
        # Here we round up to the nearest even integer.
        box_dims = []
        for length in side_lengths:
            ceil_val = int(math.ceil(length))
            if ceil_val % 2 != 0:
                ceil_val += 1
            box_dims.append(ceil_val)
        
        # Ensure a minimum box size (e.g., 8A) to avoid MT solver errors
        # if the molecule is extremely small (like a single atom/ion).
        final_dims = tuple(max(dim, 8) for dim in box_dims)
        
        return final_dims

    def compute_electronic_config(
        self, unique_elements: List[str], counts: Dict[str, int]
    ) -> Tuple[int, bool]:
        if self.method not in {"pbe", "pbemol", "scan", "tzvp"}:
            return 0, False, None
        missing = {e for e in unique_elements if e not in self.elem_data}
        if missing:
            return 0, False, missing
        total_e = sum(self.elem_data[e]["Zval"] * counts[e] for e in unique_elements)
        return total_e, (total_e % 2 == 1), None

    def _to_odd(self, n: int) -> int:
        return n + 1 - (n % 2)

    def _make_odd_min_5(self, n):
        """Ensures a k-point dimension is odd and at least 5."""
        n = int(round(n))
        # Ensure minimum of 5
        if n < 5:
            return 5
        # Ensure odd
        if n % 2 == 0:
            return n + 1
        return n

    def compute_kpoints2(self, struct, sym, bravais):
        if sym == "XYZ":
            # Generate density-based k-points
            kpts_obj = Kpoints.automatic_density(struct, kppa=KPOINTS_ACC)
            kx, ky, kz = kpts_obj.kpts[0]
            
            # Apply constraints to all dimensions in 3D
            kx = self._make_odd_min_5(kx)
            ky = self._make_odd_min_5(ky)
            kz = self._make_odd_min_5(kz)

        elif sym == "XY":
            atoms = AseAtomsAdaptor.get_atoms(struct)
            # ASE returns a list of 3 integers
            kpts = kptdensity2monkhorstpack(atoms, kptdensity=KPOINTS_DENSITY)
            
            # Apply constraints to X and Y
            kx = self._make_odd_min_5(kpts[0])
            ky = self._make_odd_min_5(kpts[1])
            # Z remains 1 for 2D/Slab calculations to avoid periodicity errors
            kz = 1
            
        return kx, ky, kz

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
            CUTOFF=600,
            KX=kx, KY=ky, KZ=kz,
            CIF_PATH=cif_path,
            KINDS=kinds_str,
            FOLDER=folder,
            REAL_SPACE="T" if self.rspace else "F",
        )
        with open(output_path, "w") as f:
            f.write(output_content)

    def write_input_scan(
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
      BASIS_SET {BASIS_FAMILY_SCAN}{q}
      POTENTIAL {GTH_FAMILY_SCAN}{q}
    &END KIND
"""
        output_content = self.template.format(
            UKS="TRUE" if unk_spins else "FALSE",
            #CUTOFF=int(max_cutoff),
            KX=kx, KY=ky, KZ=kz,
            CIF_PATH=cif_path,
            KINDS=kinds_str,
            FOLDER=folder,
            REAL_SPACE="T" if self.rspace else "F",
        )
        with open(output_path, "w") as f:
            f.write(output_content)

    def write_input_tzvp(
        self,
        output_path: str,
        cif_path: str,
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
      BASIS_SET {BASIS_FAMILY_TZVP}{q}
      POTENTIAL {GTH_FAMILY_TZVP}{q}
    &END KIND
"""
        output_content = self.template.format(
            UKS="TRUE" if unk_spins else "FALSE",
            KX=kx, KY=ky, KZ=kz,
            CIF_PATH=cif_path,
            KINDS=kinds_str,
            FOLDER=folder,
            REAL_SPACE="T" if self.rspace else "F",
        )
        with open(output_path, "w") as f:
            f.write(output_content)

    def write_input_pbemol(
        self,
        output_path: str,
        xyz_path: str,
        max_cutoff: float,
        elem_q: Dict[str, int],
        unique_elements: List[str],
        unk_spins: bool,
        folder: str,
        ax, ay, az
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
            XYZ_PATH=xyz_path,
            AX=f"{ax:.6f}",
            AY=f"{ay:.6f}",
            AZ=f"{az:.6f}",
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
        output_content = self.template.format(KX=kx, KY=ky, KZ=kz, CIF_PATH=cif_path, FOLDER=folder, PERIODIC=periodic, REAL_SPACE="T" if self.rspace else "F")
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

        if self.method in {"xyz", "pbemol"}:
            mol, unique, counts = self.load_molecule(input_path)

            # xtb does not use box size so this is purely cosmetic
            ax, ay, az = self.compute_box_size(mol, padding=VACUUM_PADDING)

            metadata["elements"] = counts
            metadata["n_atoms"] = len(mol)
            metadata["box_size"] = {"x": ax, "y": ay, "z": az}

            if self.method == "xyz":
                self.write_input_xyz(output_path, input_path, ax, ay, az, folder_rel)
            else:
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

                self.write_input_pbemol(
                    output_path, input_path, max_cutoff, elem_q, unique, unk_spins, folder_rel, ax, ay, az
                )
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
        elif self.method == "scan":
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

            self.write_input_scan(
                output_path, input_path, max_cutoff,
                kx, ky, kz, elem_q, unique, unk_spins, folder_rel
            )
        elif self.method == "tzvp":
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

            self.write_input_tzvp(
                output_path, input_path,
                kx, ky, kz, elem_q, unique, unk_spins, folder_rel
            )
        else:  # xtb
            self.write_input_xtb(output_path, input_path, kx, ky, kz, folder_rel, periodic=self.sym)
        return metadata
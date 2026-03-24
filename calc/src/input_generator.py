"""CP2K input file generator for all supported methods."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.calculator import kptdensity2monkhorstpack

from .config import KPOINTS_ACC, KPOINTS_DENSITY, VACUUM_PADDING, METHODS


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

        if self.method not in METHODS:
            raise ValueError(f"Unknown method: {method}. Must be one of {list(METHODS)}")

        self.cfg = METHODS[self.method]
        self.logger = logger or logging.getLogger(__name__)
        self.elem_data = None

        if self.cfg.elem_data_path is not None:
            with open(self.cfg.elem_data_path) as f:
                self.elem_data = json.load(f)

        with open(template_file) as f:
            self.template = f.read()

    # ── Structure / molecule loaders ─────────────────────────────────────────

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

    # ── Box / k-point helpers ────────────────────────────────────────────────

    def compute_box_size(self, mol, padding: float = VACUUM_PADDING) -> Tuple[int, int, int]:
        """Orthorhombic box for a molecule; dims rounded up to nearest even int, min 8 Å."""
        coords = mol.cart_coords
        spans = coords.max(axis=0) - coords.min(axis=0)
        side_lengths = spans + 2 * padding
        box_dims = []
        for length in side_lengths:
            ceil_val = int(math.ceil(length))
            if ceil_val % 2 != 0:
                ceil_val += 1
            box_dims.append(ceil_val)
        return tuple(max(dim, 8) for dim in box_dims)

    def _to_odd(self, n: int) -> int:
        return n + 1 - (n % 2)

    def _make_odd_min_5(self, n: int) -> int:
        n = int(round(n))
        if n < 5:
            return 5
        return n if n % 2 != 0 else n + 1

    def compute_kpoints(self, struct: Structure, sym: str) -> Tuple[int, int, int]:
        atoms = AseAtomsAdaptor.get_atoms(struct)
        kpts = kptdensity2monkhorstpack(atoms, kptdensity=KPOINTS_DENSITY)
        kx, ky, kz = int(kpts[0]), int(kpts[1]), int(kpts[2])
        if sym == "3D":
            return self._to_odd(kx), self._to_odd(ky), self._to_odd(kz)
        else:  # 2D
            return self._to_odd(kx), self._to_odd(ky), 1

    # ── Electronic configuration ─────────────────────────────────────────────

    def compute_electronic_config(
        self, unique_elements: List[str], counts: Dict[str, int]
    ) -> Tuple[int, bool, set]:
        """Returns (total_valence_e, is_odd_electron, missing_elements)."""
        missing = {e for e in unique_elements if e not in self.elem_data}
        if missing:
            return 0, False, missing
        total_e = sum(self.elem_data[e]["Zval"] * counts[e] for e in unique_elements)
        return total_e, (total_e % 2 == 1), None

    def get_max_cutoff(self, unique_elements: List[str]) -> float:
        return max(self.elem_data[e]["cutoff"] for e in unique_elements)

    def get_element_charges(self, unique_elements: List[str]) -> Dict[str, int]:
        return {e: self.elem_data[e]["Zval"] for e in unique_elements}

    # ── Template writers ─────────────────────────────────────────────────────

    def _write_input_periodic_dft(
        self,
        output_path: str,
        cif_path: str,
        kx: int, ky: int, kz: int,
        elem_q: Dict[str, int],
        unique_elements: List[str],
        unk_spins: bool,
        folder: str,
    ) -> None:
        """Unified writer for dz / scan / tzvp periodic DFT methods."""
        cfg = self.cfg
        kinds_str = "".join(
            f"\n  &KIND {e}\n    BASIS_SET {cfg.basis_family}{elem_q[e]}\n"
            f"    POTENTIAL {cfg.gth_family}{elem_q[e]}\n  &END KIND\n"
            for e in unique_elements
        )
        # Pass all possible kwargs; template.format() silently ignores extras.
        content = self.template.format(
            BASIS_SET_FILE_NAME=cfg.basis_path,
            POTENTIAL_FILE_NAME=cfg.potential_path,
            UKS="TRUE" if unk_spins else "FALSE",
            CUTOFF=600,
            KX=kx, KY=ky, KZ=kz,
            CIF_PATH=cif_path,
            KINDS=kinds_str,
            FOLDER=folder,
            REAL_SPACE="T" if self.rspace else "F",
        )
        Path(output_path).write_text(content)

    def _write_input_mol_dft(
        self,
        output_path: str,
        xyz_path: str,
        max_cutoff: float,
        elem_q: Dict[str, int],
        unique_elements: List[str],
        unk_spins: bool,
        folder: str,
        ax: float, ay: float, az: float,
    ) -> None:
        """Writer for dz_mol (molecular DFT)."""
        cfg = self.cfg
        kinds_str = "".join(
            f"\n  &KIND {e}\n    BASIS_SET {cfg.basis_family}{elem_q[e]}\n"
            f"    POTENTIAL {cfg.gth_family}{elem_q[e]}\n  &END KIND\n"
            for e in unique_elements
        )
        content = self.template.format(
            BASIS_SET_FILE_NAME=cfg.basis_path,
            POTENTIAL_FILE_NAME=cfg.potential_path,
            UKS="TRUE" if unk_spins else "FALSE",
            CUTOFF=int(max_cutoff),
            XYZ_PATH=xyz_path,
            AX=f"{ax:.6f}", AY=f"{ay:.6f}", AZ=f"{az:.6f}",
            KINDS=kinds_str,
            FOLDER=folder,
        )
        Path(output_path).write_text(content)

    _DIM_TO_CP2K = {"3D": "XYZ", "2D": "XY"}

    def _write_input_xtb(
        self,
        output_path: str,
        cif_path: str,
        kx: int, ky: int, kz: int,
        folder: str,
        periodic: str,
    ) -> None:
        """Writer for xtb (periodic xTB)."""
        content = self.template.format(
            KX=kx, KY=ky, KZ=kz,
            CIF_PATH=cif_path,
            FOLDER=folder,
            PERIODIC=self._DIM_TO_CP2K.get(periodic, periodic),
            REAL_SPACE="T" if self.rspace else "F",
        )
        Path(output_path).write_text(content)

    def _write_input_xtb_mol(
        self,
        output_path: str,
        xyz_path: str,
        ax: float, ay: float, az: float,
        folder: str,
    ) -> None:
        """Writer for xtb_mol (molecular xTB)."""
        content = self.template.format(
            XYZ_PATH=xyz_path,
            AX=f"{ax:.6f}", AY=f"{ay:.6f}", AZ=f"{az:.6f}",
            FOLDER=folder,
        )
        Path(output_path).write_text(content)

    # ── Main entry point ─────────────────────────────────────────────────────

    def generate_input(
        self,
        input_path: str,
        output_path: str,
        band: bool = False,
        uks: bool = False,
    ) -> Dict:
        """Generate a CP2K input file. Returns a metadata dict."""
        cfg = self.cfg
        metadata: Dict = {"skipped": False}
        folder_rel = "matrices"
        (Path(output_path).parent / folder_rel).mkdir(exist_ok=True)

        # ── Molecular branch ─────────────────────────────────────────────────
        if not cfg.periodic:
            mol, unique, counts = self.load_molecule(input_path)
            ax, ay, az = self.compute_box_size(mol)
            metadata.update({
                "elements": counts,
                "n_atoms": len(mol),
                "box_size": {"x": ax, "y": ay, "z": az},
            })

            if cfg.has_dft:
                total_e, unk_spins, missing = self.compute_electronic_config(unique, counts)
                if missing:
                    metadata["skipped"] = True
                    metadata["skip_reason"] = f"Missing elems: {', '.join(sorted(missing))}"
                    return metadata
                unk_spins = uks or unk_spins
                metadata.update({"total_valence_electrons": total_e, "unrestricted": unk_spins})
                max_cutoff = self.get_max_cutoff(unique)
                elem_q = self.get_element_charges(unique)
                metadata.update({"cutoff": max_cutoff, "element_charges": elem_q})
                self._write_input_mol_dft(
                    output_path, input_path, max_cutoff, elem_q, unique,
                    unk_spins, folder_rel, ax, ay, az,
                )
            else:
                self._write_input_xtb_mol(output_path, input_path, ax, ay, az, folder_rel)
            return metadata

        # ── Periodic branch ──────────────────────────────────────────────────
        struct, unique, counts = self.load_structure(input_path)
        kx, ky, kz = self.compute_kpoints(struct, self.sym)
        metadata.update({"kpoints": {"x": kx, "y": ky, "z": kz}, "elements": counts})

        if cfg.has_dft:
            total_e, unk_spins, missing = self.compute_electronic_config(unique, counts)
            if missing:
                metadata["skipped"] = True
                metadata["skip_reason"] = f"Missing elems: {', '.join(sorted(missing))}"
                return metadata
            unk_spins = uks or unk_spins
            metadata.update({"total_valence_electrons": total_e, "unrestricted": unk_spins})
            max_cutoff = self.get_max_cutoff(unique)
            elem_q = self.get_element_charges(unique)

            # Band setup: recompute k-path and elem config for the band structure cell.
            # NOTE: the input CIF should already be a standardised primitive cell.
            if band and cfg.supports_band:
                from .bands import setup_band_calculation, build_band_block, insert_band_block_into_inp
                struct_band, segments = setup_band_calculation(input_path)
                elements_band = [str(site.specie) for site in struct_band.sites]
                unique = sorted(set(elements_band))
                counts = {e: elements_band.count(e) for e in unique}
                total_e, unk_spins, _ = self.compute_electronic_config(unique, counts)
                elem_q = self.get_element_charges(unique)
                max_cutoff = self.get_max_cutoff(unique)
                kx, ky, kz = self.compute_kpoints(struct_band, self.sym)
                metadata["band"] = True

            metadata.update({"cutoff": max_cutoff, "element_charges": elem_q})
            self._write_input_periodic_dft(
                output_path, input_path, kx, ky, kz, elem_q, unique, unk_spins, folder_rel,
            )

            if band and cfg.supports_band:
                inp_text = Path(output_path).read_text()
                # ADDED_MOS -1 triggers CP2K bug (qs_environment.F:1695) with UKS; use 20
                Path(output_path).write_text(
                    insert_band_block_into_inp(inp_text, build_band_block(segments, added_mos=20))
                )
        else:
            self._write_input_xtb(output_path, input_path, kx, ky, kz, folder_rel, self.sym)

        return metadata

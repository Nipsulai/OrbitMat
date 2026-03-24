"""Band structure generation, parsing, and plotting for CP2K."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.dft.kpoints import resolve_kpt_path_string, kpoint_convert

LINE_DENSITY = 100   # k-points per inverse Angstrom (matches FHI-AIMS database)
MIN_KPOINTS  = 20    # CP2K requires NPOINTS >= 1; FHI-AIMS has no minimum
BAND_FILENAME = "BAND"

# ── primitive cell ────────────────────────────────────────────────────────────

def _fix_label(label: str) -> str:
    return "GAMMA" if label in (r"\Gamma", "Γ", "GAMMA", "G") else label


def _segments_from_bandpath(atoms, line_density: int, min_kpoints: int) -> list:
    """Build CP2K KPOINT_SET segments from ASE bandpath (FHI-AIMS convention)."""
    bp = atoms.cell.bandpath()
    r_kpts = resolve_kpt_path_string(bp.path, bp.special_points)
    segments = []
    for labels, coords in zip(*r_kpts):
        dists   = coords[1:] - coords[:-1]
        lengths = [np.linalg.norm(d) for d in kpoint_convert(atoms.cell, skpts_kc=dists)]
        for (la, lb), start, end, length in zip(
            zip(labels[:-1], labels[1:]), coords[:-1], coords[1:], lengths
        ):
            npoints = max(min_kpoints, int(round(length * line_density)))
            segments.append((
                [(_fix_label(la), start.tolist()), (_fix_label(lb), end.tolist())],
                npoints,
            ))
    return segments


def setup_band_calculation(cif_path: str,
                            line_density: int = LINE_DENSITY,
                            min_kpoints: int = MIN_KPOINTS,
                            ) -> tuple[Structure, list]:
    """
    Band setup using ASE bandpath (same convention as FHI-AIMS database).
    Returns (struct, segments).
    """
    struct = Structure.from_file(cif_path)
    atoms  = AseAtomsAdaptor.get_atoms(struct)
    segments = _segments_from_bandpath(atoms, line_density, min_kpoints)
    return struct, segments


# keep for backward compat / standalone use
def standardize_primitive(cif_path: str) -> Structure:
    prim, _ = setup_band_calculation(cif_path)
    return prim


# ── k-path ────────────────────────────────────────────────────────────────────

def get_kpath_segments(struct: Structure, line_density: int = LINE_DENSITY,
                       min_kpoints: int = MIN_KPOINTS) -> list:
    atoms = AseAtomsAdaptor.get_atoms(struct)
    return _segments_from_bandpath(atoms, line_density, min_kpoints)


# ── CP2K block builder ────────────────────────────────────────────────────────

def build_band_block(segments: list, added_mos: int = -1,
                     filename: str = BAND_FILENAME) -> str:
    i4, i6, i8 = "    ", "      ", "        "
    lines = [f"{i4}&BAND_STRUCTURE",
             f"{i6}ADDED_MOS {added_mos}",
             f"{i6}FILE_NAME {filename}"]
    for seg_pts, npoints in segments:
        lines.append(f"{i6}&KPOINT_SET")
        lines.append(f"{i8}UNITS B_VECTOR")
        for label, (kx, ky, kz) in seg_pts:
            lines.append(f"{i8}SPECIAL_POINT {label:<8} {kx:10.6f} {ky:10.6f} {kz:10.6f}")
        lines.append(f"{i8}NPOINTS {npoints}")
        lines.append(f"{i6}&END KPOINT_SET")
    lines.append(f"{i4}&END BAND_STRUCTURE")
    return "\n".join(lines) + "\n"


def insert_band_block_into_inp(inp_text: str, block: str) -> str:
    """Insert band block just before &END PRINT inside &DFT."""
    dft_end = inp_text.rfind("&END DFT")
    if dft_end == -1:
        raise ValueError("Could not find &END DFT in input file")
    dft_section = inp_text[:dft_end]
    end_print_pos = dft_section.rfind("&END PRINT")
    if end_print_pos == -1:
        raise ValueError("Could not find &END PRINT inside &DFT")
    line_start = inp_text.rfind("\n", 0, end_print_pos) + 1
    return inp_text[:line_start] + block + "\n" + inp_text[line_start:]


# ── band file parser ──────────────────────────────────────────────────────────

@dataclass
class BandSet:
    labels: list
    kfrac:  list
    bands:  np.ndarray   # (n_kpts, n_bands) energies in eV
    occs:   np.ndarray   # (n_kpts, n_bands) occupations


def parse_band_file(path: str) -> list[BandSet]:
    sets = []
    current_labels, current_kfrac = [], []
    current_energies, current_occs = [], []
    in_kpoint = False
    kpoint_energies, kpoint_occs = [], []
    kpoint_coord = None

    set_re     = re.compile(r"^# Set \d+:")
    special_re = re.compile(r"^#\s+Special point \d+\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+(\S+)")
    point_re   = re.compile(r"^#\s+Point \d+\s+Spin \d+:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)")
    energy_re  = re.compile(r"^\s+\d+\s+([\d.\-]+)\s+([\d.\-]+)")

    def _flush_kpoint():
        nonlocal kpoint_energies, kpoint_occs, kpoint_coord, in_kpoint
        if in_kpoint and kpoint_coord is not None:
            current_kfrac.append(kpoint_coord)
            current_energies.append(kpoint_energies[:])
            current_occs.append(kpoint_occs[:])
        kpoint_energies.clear(); kpoint_occs.clear()
        kpoint_coord = None; in_kpoint = False

    def _flush_set():
        nonlocal current_labels, current_kfrac, current_energies, current_occs
        if current_energies:
            sets.append(BandSet(
                labels=current_labels[:],
                kfrac=current_kfrac[:],
                bands=np.array(current_energies),
                occs=np.array(current_occs),
            ))
        current_labels.clear(); current_kfrac.clear()
        current_energies.clear(); current_occs.clear()

    with open(path) as f:
        for line in f:
            if set_re.match(line):
                _flush_kpoint(); _flush_set(); continue
            m = special_re.match(line)
            if m:
                lbl = m.group(4)
                current_labels.append("G" if lbl in ("GAMMA", r"\Gamma") else lbl)
                continue
            m = point_re.match(line)
            if m:
                _flush_kpoint()
                kpoint_coord = np.array([float(m.group(i)) for i in (1, 2, 3)])
                in_kpoint = True; continue
            m = energy_re.match(line)
            if m and in_kpoint:
                kpoint_energies.append(float(m.group(1)))
                kpoint_occs.append(float(m.group(2)))

    _flush_kpoint(); _flush_set()
    return sets


# ── x-axis ────────────────────────────────────────────────────────────────────

def find_vbm(band_sets: list[BandSet]) -> float:
    """Highest energy band with occupation > 0.5 across all k-points."""
    vbm = -np.inf
    for bs in band_sets:
        occ = bs.bands[bs.occs > 0.5]
        if occ.size:
            vbm = max(vbm, float(occ.max()))
    return vbm


def build_xaxis(band_sets: list[BandSet], recip_lattice):
    def _dist(a, b):
        return float(np.linalg.norm(
            recip_lattice.get_cartesian_coords(b) - recip_lattice.get_cartesian_coords(a)
        ))

    x_per_set, ticks_x, ticks_lbl, disc_x = [], [], [], []
    cursor = 0.0

    for i, bs in enumerate(band_sets):
        n = len(bs.kfrac)
        if not n:
            continue
        if i > 0 and band_sets[i - 1].labels[-1] != bs.labels[0]:
            disc_x.append(cursor)

        xs = np.zeros(n)
        xs[0] = cursor
        for j in range(1, n):
            xs[j] = xs[j - 1] + _dist(bs.kfrac[j - 1], bs.kfrac[j])

        if not ticks_x or abs(ticks_x[-1] - xs[0]) > 1e-8:
            ticks_x.append(xs[0])
            ticks_lbl.append(bs.labels[0])
        ticks_x.append(xs[-1])
        ticks_lbl.append(bs.labels[-1])
        x_per_set.append(xs)
        cursor = xs[-1]

    # Merge duplicate ticks
    merged_x, merged_lbl = [], []
    for x, l in zip(ticks_x, ticks_lbl):
        if merged_x and abs(merged_x[-1] - x) < 1e-8:
            if merged_lbl[-1] != l:
                merged_lbl[-1] = f"{merged_lbl[-1]}|{l}"
        else:
            merged_x.append(x); merged_lbl.append(l)

    return x_per_set, merged_x, merged_lbl, disc_x


# ── plot ──────────────────────────────────────────────────────────────────────

def save_band_plot(band_sets: list[BandSet], recip_lattice, out_png: str,
                   emin: float = -6.0, emax: float = 6.0,
                   title: str = "") -> dict:
    """
    Plot band structure and save to out_png.
    Returns dict with vbm and bandgap (min direct/indirect gap above VBM).
    """
    x_per_set, ticks_x, ticks_lbl, disc_x = build_xaxis(band_sets, recip_lattice)
    vbm = find_vbm(band_sets)

    # Estimate band gap: lowest unoccupied - VBM
    cbe_vals = []
    for bs in band_sets:
        unocc = bs.bands[bs.occs < 0.5]
        if unocc.size:
            cbe_vals.append(float(unocc.min()))
    cbm = min(cbe_vals) if cbe_vals else None
    bandgap = round(max(0.0, cbm - vbm), 4) if cbm is not None else None

    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": 0.6, "axes.grid": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 150, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    lw = 0.6
    for xs, bs in zip(x_per_set, band_sets):
        energies = bs.bands - vbm
        for b in range(energies.shape[1]):
            ax.plot(xs, energies[:, b], color="#2B7BB9", lw=lw, rasterized=True)

    for x in ticks_x:
        ax.axvline(x, color="k", lw=0.5, zorder=3)
    for x in disc_x:
        ax.axvline(x, color="k", lw=1.2, zorder=4)
    ax.axhline(0, color="k", lw=0.5, ls="--", zorder=2)

    ax.set_xlim(ticks_x[0], ticks_x[-1])
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_lbl)
    ax.set_ylim(emin, emax)
    ax.set_ylabel("E - E_VBM (eV)")
    ax.tick_params(axis="x", bottom=False, top=False)
    gap_str = f"  |  Gap = {bandgap:.3f} eV" if bandgap is not None else ""
    ax.set_title((title or "") + gap_str)

    fig.savefig(out_png)
    plt.close(fig)

    return {"vbm_ev": vbm, "bandgap_ev": bandgap}


def postproc_bands(work_dir: str, prim_cif: str, matrices_pkl: str,
                   emin: float = -6.0, emax: float = 6.0) -> dict:
    """
    Parse CP2K BAND file, add band_structure and bandgap_path_eV to
    matrices.pkl (top-level), save bands.png. Deletes BAND file after parsing.
    Returns dict with vbm_ev and bandgap_path_eV.
    """
    import pickle
    work_dir = Path(work_dir)
    band_file = work_dir / BAND_FILENAME
    if not band_file.exists():
        raise FileNotFoundError(f"BAND file not found: {band_file}")

    prim = Structure.from_file(prim_cif)
    recip = prim.lattice.reciprocal_lattice
    formula = prim.composition.reduced_formula
    spg = prim.get_space_group_info()
    title = f"{formula}  SG {spg[1]} ({spg[0]})"

    band_sets = parse_band_file(str(band_file))

    x_per_set, ticks_x, ticks_lbl, disc_x = build_xaxis(band_sets, recip)
    vbm = find_vbm(band_sets)

    out_png = str(work_dir / "bands.png")
    stats = save_band_plot(band_sets, recip, out_png,
                           emin=emin, emax=emax, title=title)

    bandgap_path = stats["bandgap_ev"]

    # Compact band structure data for storage
    band_data = {
        "x_per_set":      [xs.tolist() for xs in x_per_set],
        "ticks_x":        ticks_x,
        "ticks_lbl":      ticks_lbl,
        "disc_x":         disc_x,
        "vbm_ev":         vbm,
        "bandgap_path_eV": bandgap_path,
        "sets": [
            {
                "labels": bs.labels,
                "bands":  bs.bands.tolist(),
                "occs":   bs.occs.tolist(),
            }
            for bs in band_sets
        ],
    }

    pkl_path = Path(matrices_pkl)
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        cbm = round(vbm + bandgap_path, 6) if bandgap_path is not None else None
        data["band_structure"]  = band_data
        data["bandgap_cp2k"] = bandgap_path
        data["vbm_cp2k_eV"]     = round(vbm, 6)
        data["cbm_cp2k_eV"]     = cbm
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

    return {"vbm_ev": vbm, "bandgap_path_eV": bandgap_path}

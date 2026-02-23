import json
import glob

from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def parse_json_structure(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    result = {
        "filename": json_path.name,
        "icsd": data["ICSD_number"],
        "snumat_id": data["SNUMAT_id"],
        "space_group_num": data["Space_group_rlx"],
        "bandgap_pbe": data["Band_gap_GGA"],
        "bandgap_hse": data["Band_gap_HSE"],
        "gap_type_pbe": data["Direct_or_indirect"],
        "gap_type_hse": data["Direct_or_indirect_HSE"],
        "magnetic_ordering": data["Magnetic_ordering"],
        "soc": data["SOC"],
    }

    poscar_str = data["Structure_rlx"]
    poscar = Poscar.from_str(poscar_str)
    structure = poscar.structure

    result["structure"] = structure
    result["n_atoms_json"] = len(structure)
    result["volume_json"] = structure.volume
    return result


def reduce_to_primitive(struct, symprec, angle_tol):
    sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
    prim = sga.get_primitive_standard_structure()
    reduced = (len(prim) < len(struct))
    return prim, reduced


def build_metadata_entry(row, struct_json, cif_path, if_reducable, multiplicity):
    # ORIGINAL structure info
    a, b, c = struct_json.lattice.angles
    orig_info = {
        "n_atoms": len(struct_json),
        "volume": struct_json.volume,
        "angles": {"alpha": a, "beta": b, "gamma": c}
    }

    # Band gaps
    bandgap_info = {
        "bandgap_pbe": row["bandgap_pbe"],
        "bandgap_hse": row["bandgap_hse"],
        "gap_type_pbe": row["gap_type_pbe"],
        "gap_type_hse": row["gap_type_hse"],
    }

    # Primitive reduction
    prim_info = {
        "if_reducable": if_reducable,
        "multiplicity": multiplicity,
    }

    brav = space_group_to_bravais(row["space_group_num"])
    bravais_info = {
        "center": brav[0],
        "lattice": brav[1], 
        "space_group": row["space_group_num"]
    }

    # Final combined metadata
    return {
        "filename_json": row["filename"],
        "cif_path": str(cif_path),
        "icsd": row["icsd"],
        "snumat_id": row["snumat_id"],
        "magnetic_ordering": row["magnetic_ordering"],
        "soc": row["soc"],
        "bandgap_info": bandgap_info,
        "orig": orig_info,
        "prim": prim_info,
        "bravais": bravais_info
    }

def space_group_to_bravais(space_group_number):
    """
    Convert space group number (1-230) to Bravais lattice type.
    
    Returns:
    --------
    tuple : (str, str)
        (centering, lattice_system) where:
        - centering: "simple", "body", "face", "base", or "0" (for hexagonal/rhombohedral)
        - lattice_system: "triclinic", "monoclinic", "orthorhombic", "tetragonal", 
                         "hexagonal", "rhombohedral", "cubic"
    
    Reference:
    ----------
    https://lampz.tugraz.at/~hadley/ss1/crystalstructure/sg2bravais.html
    """
    
    # Mapping from space group number to (centering, lattice_system)
    sg_to_bravais = {
        # Triclinic (1-2)
        1: ("simple", "triclinic"),
        2: ("simple", "triclinic"),
        
        # Monoclinic (3-15)
        3: ("simple", "monoclinic"),
        4: ("simple", "monoclinic"),
        5: ("base", "monoclinic"),
        6: ("simple", "monoclinic"),
        7: ("simple", "monoclinic"),
        8: ("base", "monoclinic"),
        9: ("base", "monoclinic"),
        10: ("simple", "monoclinic"),
        11: ("simple", "monoclinic"),
        12: ("base", "monoclinic"),
        13: ("simple", "monoclinic"),
        14: ("simple", "monoclinic"),
        15: ("base", "monoclinic"),
        
        # Orthorhombic (16-74)
        16: ("simple", "orthorhombic"),
        17: ("simple", "orthorhombic"),
        18: ("simple", "orthorhombic"),
        19: ("simple", "orthorhombic"),
        20: ("base", "orthorhombic"),
        21: ("base", "orthorhombic"),
        22: ("face", "orthorhombic"),
        23: ("body", "orthorhombic"),
        24: ("body", "orthorhombic"),
        25: ("simple", "orthorhombic"),
        26: ("simple", "orthorhombic"),
        27: ("simple", "orthorhombic"),
        28: ("simple", "orthorhombic"),
        29: ("simple", "orthorhombic"),
        30: ("simple", "orthorhombic"),
        31: ("simple", "orthorhombic"),
        32: ("simple", "orthorhombic"),
        33: ("simple", "orthorhombic"),
        34: ("simple", "orthorhombic"),
        35: ("base", "orthorhombic"),
        36: ("base", "orthorhombic"),
        37: ("base", "orthorhombic"),
        38: ("base", "orthorhombic"),
        39: ("base", "orthorhombic"),
        40: ("base", "orthorhombic"),
        41: ("base", "orthorhombic"),
        42: ("face", "orthorhombic"),
        43: ("face", "orthorhombic"),
        44: ("body", "orthorhombic"),
        45: ("body", "orthorhombic"),
        46: ("body", "orthorhombic"),
        47: ("simple", "orthorhombic"),
        48: ("simple", "orthorhombic"),
        49: ("simple", "orthorhombic"),
        50: ("simple", "orthorhombic"),
        51: ("simple", "orthorhombic"),
        52: ("simple", "orthorhombic"),
        53: ("simple", "orthorhombic"),
        54: ("simple", "orthorhombic"),
        55: ("simple", "orthorhombic"),
        56: ("simple", "orthorhombic"),
        57: ("simple", "orthorhombic"),
        58: ("simple", "orthorhombic"),
        59: ("simple", "orthorhombic"),
        60: ("simple", "orthorhombic"),
        61: ("simple", "orthorhombic"),
        62: ("simple", "orthorhombic"),
        63: ("base", "orthorhombic"),
        64: ("base", "orthorhombic"),
        65: ("base", "orthorhombic"),
        66: ("base", "orthorhombic"),
        67: ("base", "orthorhombic"),
        68: ("base", "orthorhombic"),
        69: ("face", "orthorhombic"),
        70: ("face", "orthorhombic"),
        71: ("body", "orthorhombic"),
        72: ("body", "orthorhombic"),
        73: ("body", "orthorhombic"),
        74: ("body", "orthorhombic"),
        
        # Tetragonal (75-142)
        75: ("simple", "tetragonal"),
        76: ("simple", "tetragonal"),
        77: ("simple", "tetragonal"),
        78: ("simple", "tetragonal"),
        79: ("body", "tetragonal"),
        80: ("body", "tetragonal"),
        81: ("simple", "tetragonal"),
        82: ("body", "tetragonal"),
        83: ("simple", "tetragonal"),
        84: ("simple", "tetragonal"),
        85: ("simple", "tetragonal"),
        86: ("simple", "tetragonal"),
        87: ("body", "tetragonal"),
        88: ("body", "tetragonal"),
        89: ("simple", "tetragonal"),
        90: ("simple", "tetragonal"),
        91: ("simple", "tetragonal"),
        92: ("simple", "tetragonal"),
        93: ("simple", "tetragonal"),
        94: ("simple", "tetragonal"),
        95: ("simple", "tetragonal"),
        96: ("simple", "tetragonal"),
        97: ("body", "tetragonal"),
        98: ("body", "tetragonal"),
        99: ("simple", "tetragonal"),
        100: ("simple", "tetragonal"),
        101: ("simple", "tetragonal"),
        102: ("simple", "tetragonal"),
        103: ("simple", "tetragonal"),
        104: ("simple", "tetragonal"),
        105: ("simple", "tetragonal"),
        106: ("simple", "tetragonal"),
        107: ("body", "tetragonal"),
        108: ("body", "tetragonal"),
        109: ("body", "tetragonal"),
        110: ("body", "tetragonal"),
        111: ("simple", "tetragonal"),
        112: ("simple", "tetragonal"),
        113: ("simple", "tetragonal"),
        114: ("simple", "tetragonal"),
        115: ("simple", "tetragonal"),
        116: ("simple", "tetragonal"),
        117: ("simple", "tetragonal"),
        118: ("simple", "tetragonal"),
        119: ("body", "tetragonal"),
        120: ("body", "tetragonal"),
        121: ("body", "tetragonal"),
        122: ("body", "tetragonal"),
        123: ("simple", "tetragonal"),
        124: ("simple", "tetragonal"),
        125: ("simple", "tetragonal"),
        126: ("simple", "tetragonal"),
        127: ("simple", "tetragonal"),
        128: ("simple", "tetragonal"),
        129: ("simple", "tetragonal"),
        130: ("simple", "tetragonal"),
        131: ("simple", "tetragonal"),
        132: ("simple", "tetragonal"),
        133: ("simple", "tetragonal"),
        134: ("simple", "tetragonal"),
        135: ("simple", "tetragonal"),
        136: ("simple", "tetragonal"),
        137: ("simple", "tetragonal"),
        138: ("simple", "tetragonal"),
        139: ("body", "tetragonal"),
        140: ("body", "tetragonal"),
        141: ("body", "tetragonal"),
        142: ("body", "tetragonal"),
        
        # Trigonal/Hexagonal (143-167)
        143: ("0", "hexagonal"),
        144: ("0", "hexagonal"),
        145: ("0", "hexagonal"),
        146: ("0", "rhombohedral"),
        147: ("0", "hexagonal"),
        148: ("0", "rhombohedral"),
        149: ("0", "hexagonal"),
        150: ("0", "hexagonal"),
        151: ("0", "hexagonal"),
        152: ("0", "hexagonal"),
        153: ("0", "hexagonal"),
        154: ("0", "hexagonal"),
        155: ("0", "rhombohedral"),
        156: ("0", "hexagonal"),
        157: ("0", "hexagonal"),
        158: ("0", "hexagonal"),
        159: ("0", "hexagonal"),
        160: ("0", "rhombohedral"),
        161: ("0", "rhombohedral"),
        162: ("0", "hexagonal"),
        163: ("0", "hexagonal"),
        164: ("0", "hexagonal"),
        165: ("0", "hexagonal"),
        166: ("0", "rhombohedral"),
        167: ("0", "rhombohedral"),
        
        # Hexagonal (168-194)
        168: ("0", "hexagonal"),
        169: ("0", "hexagonal"),
        170: ("0", "hexagonal"),
        171: ("0", "hexagonal"),
        172: ("0", "hexagonal"),
        173: ("0", "hexagonal"),
        174: ("0", "hexagonal"),
        175: ("0", "hexagonal"),
        176: ("0", "hexagonal"),
        177: ("0", "hexagonal"),
        178: ("0", "hexagonal"),
        179: ("0", "hexagonal"),
        180: ("0", "hexagonal"),
        181: ("0", "hexagonal"),
        182: ("0", "hexagonal"),
        183: ("0", "hexagonal"),
        184: ("0", "hexagonal"),
        185: ("0", "hexagonal"),
        186: ("0", "hexagonal"),
        187: ("0", "hexagonal"),
        188: ("0", "hexagonal"),
        189: ("0", "hexagonal"),
        190: ("0", "hexagonal"),
        191: ("0", "hexagonal"),
        192: ("0", "hexagonal"),
        193: ("0", "hexagonal"),
        194: ("0", "hexagonal"),
        
        # Cubic (195-230)
        195: ("simple", "cubic"),
        196: ("face", "cubic"),
        197: ("body", "cubic"),
        198: ("simple", "cubic"),
        199: ("body", "cubic"),
        200: ("simple", "cubic"),
        201: ("simple", "cubic"),
        202: ("face", "cubic"),
        203: ("face", "cubic"),
        204: ("body", "cubic"),
        205: ("simple", "cubic"),
        206: ("body", "cubic"),
        207: ("simple", "cubic"),
        208: ("simple", "cubic"),
        209: ("face", "cubic"),
        210: ("face", "cubic"),
        211: ("body", "cubic"),
        212: ("simple", "cubic"),
        213: ("simple", "cubic"),
        214: ("body", "cubic"),
        215: ("simple", "cubic"),
        216: ("face", "cubic"),
        217: ("body", "cubic"),
        218: ("simple", "cubic"),
        219: ("face", "cubic"),
        220: ("body", "cubic"),
        221: ("simple", "cubic"),
        222: ("simple", "cubic"),
        223: ("simple", "cubic"),
        224: ("simple", "cubic"),
        225: ("face", "cubic"),
        226: ("face", "cubic"),
        227: ("face", "cubic"),
        228: ("face", "cubic"),
        229: ("body", "cubic"),
        230: ("body", "cubic"),
    }
    
    if space_group_number < 1 or space_group_number > 230:
        raise ValueError("Space group number must be between 1 and 230")
    
    return sg_to_bravais[space_group_number]

def find_path(N, json=False):
    if json:
        matches = glob.glob(f"/home/nikolai/OrbMat/data/bandgap2020/*_{N}.json")
    else:
        matches = glob.glob(f"/home/nikolai/OrbMat/data/cifs/prim/*_{N}.cif")
    
    if not matches:
        print("No files found.")
    else:
        for path in matches:
            print(path)

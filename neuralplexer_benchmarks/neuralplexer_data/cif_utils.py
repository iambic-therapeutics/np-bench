import json
from collections import defaultdict
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from parmed import Structure, load_file
from parmed.formats.pdb import _is_hetatm
from parmed.formats.pdbx import PdbxContainers, PdbxReader
from rdkit import Chem

from neuralplexer_benchmarks.neuralplexer_data.constants import (
    ARTIFACT_RESNAMES,
    CIF_BOND_TYPE_TO_RDKIT_BOND_TYPE,
    CRYSTALLIZATION_AIDS,
    ION_RESNAMES,
    PARMED_EXPANDED_RESIDUE_TEMPLATE_MATCH_RESIDUES,
    SOLVENT_RESNAMES,
)
from neuralplexer_benchmarks.neuralplexer_data.datamodels import (
    POLYMER_TYPE_MAP,
    AtomIdentifier,
    BondType,
    Branch,
    ExplicitBond,
    IsExperimentallyResolvedDict,
    NPLXV3Input,
    PolymerType,
    PolymerTypeDict,
    ResidueId,
    SequenceResidue,
    Sequences,
    validate_nplx_v3_input,
)
from neuralplexer_benchmarks.neuralplexer_data.exceptions import (
    BondError,
    CorruptedDataError,
    NPLXV3Error,
)
from neuralplexer_benchmarks.neuralplexer_data.smiles_utils import get_smiles_for_resname

LOGGER = getLogger(__name__)

RESNAME_TO_TYPE_PATH = Path(__file__).parent / "resources" / "resname_to_type.json"


@cache
def get_resname_to_type_map() -> dict[str, str]:
    return json.loads(RESNAME_TO_TYPE_PATH.read_text())


def load_mm_cif(
    cif_path: str,
    *,
    all_residue_template_match: bool = False,
    read_chem_comp_bonds: bool = True,
    is_crystal: bool = False,
) -> NPLXV3Input:
    container = get_cif_container(cif_path)
    structure = load_file(
        str(cif_path),
        expanded_residue_template_match=True,
        all_residue_template_match=all_residue_template_match,
    )
    result = parse_structure(structure, container, is_crystal)

    # Explicitly parse struct_conn records, because parmed gets the bond types wrong
    atom_identifiers = result.atom_identifiers
    explicit_bonds = _extract_explicit_bonds(container)
    for explicit_bond in explicit_bonds:
        i = atom_identifiers.index(explicit_bond.from_atom)
        j = atom_identifiers.index(explicit_bond.to_atom)
        result.bonds[
            (result.bonds[:, 0] == i) & (result.bonds[:, 1] == j)
            | (result.bonds[:, 0] == j) & (result.bonds[:, 1] == i),
            2,
        ] = explicit_bond.bond_type

    if not read_chem_comp_bonds or container.getObj("chem_comp_bond") is None:
        return result
    chem_comp_bond_data = extract_cif_data(
        container,
        "chem_comp_bond",
        [
            "comp_id",
            "atom_id_1",
            "atom_id_2",
            "value_order",
            "pdbx_aromatic_flag",
        ],
    )
    residue_names_to_chem_comp_bonds = defaultdict(list)
    for chem_comp_bond in chem_comp_bond_data:
        if str(chem_comp_bond["pdbx_aromatic_flag"]).strip().upper() == "Y":
            bond_type = BondType.AROMATIC
        else:
            bond_type = _value_order_to_bond_type(chem_comp_bond["value_order"], chem_comp_bond["pdbx_aromatic_flag"])

        residue_names_to_chem_comp_bonds[chem_comp_bond["comp_id"]].append(
            (
                chem_comp_bond["atom_id_1"],
                chem_comp_bond["atom_id_2"],
                bond_type,
            )
        )
    additional_bonds = []
    for i, (residue_name, atom_identifier) in enumerate(zip(result.residue_name, atom_identifiers)):
        if (
            residue_name in PARMED_EXPANDED_RESIDUE_TEMPLATE_MATCH_RESIDUES
            or residue_name not in residue_names_to_chem_comp_bonds
        ):
            continue
        residue_id, atom_type = atom_identifier
        for atom_id_1, atom_id_2, bond_type in residue_names_to_chem_comp_bonds[residue_name]:
            if atom_type == atom_id_1 or atom_type == atom_id_2:
                for j_, other_atom_identifier in enumerate(atom_identifiers[i + 1 :]):
                    other_residue_id, other_atom_type = other_atom_identifier
                    if other_residue_id != residue_id:
                        break
                    if other_atom_type == atom_id_1 or other_atom_type == atom_id_2:
                        j = j_ + i + 1
                        i, j = min(i, j), max(i, j)
                        additional_bonds.append((i, j, int(bond_type)))
                        break

    for i, j, int_bond_type in additional_bonds:
        if not any((result.bonds[:, 0] == i) & (result.bonds[:, 1] == j)) and not any(
            (result.bonds[:, 0] == j) & (result.bonds[:, 1] == i)
        ):
            result.bonds = np.vstack([result.bonds, [i, j, int_bond_type]])
        else:
            result.bonds[
                (result.bonds[:, 0] == i) & (result.bonds[:, 1] == j)
                | (result.bonds[:, 0] == j) & (result.bonds[:, 1] == i),
                2,
            ] = int_bond_type

    return result


def _value_order_to_bond_type(value_order: str, pdbx_aromatic_flag: str) -> BondType:
    if str(pdbx_aromatic_flag).strip().upper() == "Y":
        return BondType.AROMATIC
    value_order = value_order.lower()
    if value_order == "sing":
        return BondType.SINGLE
    elif value_order == "doub":
        return BondType.DOUBLE
    elif value_order == "trip":
        return BondType.TRIPLE
    elif value_order == "quad":
        return BondType.QUADRUPLE
    elif value_order == "arom":
        return BondType.AROMATIC
    else:
        raise ValueError(f"Unsupported bond type: {value_order}")


def get_cif_container(cif_path: Path | str) -> PdbxContainers.ContainerBase:
    data: list[Any] = []
    with open(cif_path) as file:
        PdbxReader(file).read(data)

    if len(data) != 1:
        raise CorruptedDataError("Expecting single structure in cif file.")

    return data[0]


def extract_cif_data(container: PdbxContainers.ContainerBase, obj_name: str, keys: list[str]) -> list[dict[str, Any]]:
    obj = container.getObj(obj_name)
    if obj is None:
        raise AttributeError(f"Object {obj_name} not found in cif file.")
    idxs: dict[str, int] = {key: obj.getAttributeIndex(key) for key in keys}
    result = []
    for i in range(obj.getRowCount()):
        row = obj.getRow(i)
        result.append({key: row[idxs[key]] if idxs[key] != -1 else None for key in keys})
    return result


class _SequenceExtractionResult(NamedTuple):
    chain_sequences: Sequences
    is_experimentally_resolved: IsExperimentallyResolvedDict
    polymer_type: PolymerTypeDict


def parse_structure(
    structure: Structure,
    container: PdbxContainers.ContainerBase,
    is_crystal: bool,
    validate: bool = True,
) -> NPLXV3Input:
    sequence_extraction_result = _extract_chain_sequences(container)

    chain_sequences = sequence_extraction_result.chain_sequences
    is_experimentally_resolved = sequence_extraction_result.is_experimentally_resolved
    polymer_type = sequence_extraction_result.polymer_type

    atomic_numbers = np.array([int(atom.atomic_number) for atom in structure.atoms])
    atom_types_pdb = np.array([atom.name for atom in structure.atoms])
    atom_coordinates = np.array([[atom.xx, atom.xy, atom.xz] for atom in structure.atoms])
    chain_id = np.array([atom.residue.chain for atom in structure.atoms])
    is_hetatm = np.array([_is_hetatm(atom.residue.name) for atom in structure.atoms])
    residue_id = np.array([atom.residue.number for atom in structure.atoms])
    residue_name = np.array([atom.residue.name for atom in structure.atoms])
    insertion_code = np.array([atom.residue.insertion_code.strip("?.") for atom in structure.atoms])
    b_factor = np.array([atom.bfactor for atom in structure.atoms])
    formal_charges = np.array([atom.formal_charge or 0 for atom in structure.atoms])

    rdkit_mol = structure.rdkit_mol
    bonds = np.array(_get_bond_list_from_rdkit_mol(rdkit_mol))

    is_solvent = np.array([atom.residue.name in SOLVENT_RESNAMES for atom in structure.atoms])
    is_ion = np.array([atom.residue.name in ION_RESNAMES for atom in structure.atoms])

    if is_crystal:
        is_artifact = np.array([atom.residue.name in CRYSTALLIZATION_AIDS for atom in structure.atoms])
    else:
        is_artifact = np.array([False] * len(structure.atoms))

    three_letter_to_smiles = {residue.name: get_smiles_for_resname(residue.name) for residue in structure.residues}

    nonpolymer_entities = _extract_nonpoly_entities(container, is_crystal)
    branch_entities = _extract_branch_entities(container)

    result = NPLXV3Input(
        atomic_numbers=atomic_numbers,
        atom_types_pdb=atom_types_pdb,
        atom_coordinates=atom_coordinates,
        chain_id=chain_id,
        is_hetatm=is_hetatm,
        residue_id=residue_id,
        residue_name=residue_name,
        insertion_code=insertion_code,
        b_factor=b_factor,
        bonds=bonds,
        chain_sequences=chain_sequences,
        is_experimentally_resolved=is_experimentally_resolved,
        polymer_type=polymer_type,
        nonpolymer_entities=nonpolymer_entities,
        branch_entities=branch_entities,
        is_solvent=is_solvent,
        is_ion=is_ion,
        is_artifact=is_artifact,
        three_letter_to_smiles=three_letter_to_smiles,
        formal_charges=formal_charges,
    )
    if validate:
        return validate_nplx_v3_input(result)
    return result


def _extract_chain_sequences(
    container: PdbxContainers.ContainerBase,
    use_auth_system: bool = True,
) -> _SequenceExtractionResult:
    # Should double-check with entity_poly_seq to ensure correct order and robustness
    # entity_poly_seq = extract_cif_data(container, "entity_poly_seq", ["entity_id", "num", "mon_id"])
    if use_auth_system:
        seq_num_field = "pdb_seq_num"
        chain_id_field = "pdb_strand_id"
    else:
        seq_num_field = "seq_id"
        chain_id_field = "asym_id"
    try:
        keys = ["entity_id", "mon_id", "pdb_mon_id", seq_num_field, chain_id_field]
        if use_auth_system:
            keys.append("pdb_ins_code")
        entity_poly_seq_scheme = extract_cif_data(
            container,
            "pdbx_poly_seq_scheme",
            keys,
        )
        entity_poly = extract_cif_data(container, "entity_poly", ["entity_id", "type"])
    except AttributeError:
        entity_poly_seq_scheme = []
        entity_poly = []

    entity_poly_type = {
        entity["entity_id"]: POLYMER_TYPE_MAP.get(entity["type"], PolymerType.OTHER) for entity in entity_poly
    }

    chain_sequences = defaultdict(list)
    is_experimentally_resolved_raw = defaultdict(list)
    polymer_type_dict: PolymerTypeDict = {}

    for residue_line in entity_poly_seq_scheme:
        sequence_identifier = residue_line[chain_id_field]
        pdb_chain_id = residue_line[chain_id_field]
        residue_name = residue_line["mon_id"]
        pdb_residue_name = residue_line["pdb_mon_id"]
        pdb_residue_number = residue_line[seq_num_field]
        pdb_insertion_code = residue_line["pdb_ins_code"].strip("?.") if use_auth_system else ""
        if pdb_residue_name.strip() == "?" or pdb_residue_number.strip() == "?":
            residue: ResidueId | None = None
            is_experimentally_resolved_raw[sequence_identifier].append(False)
        else:
            residue = ResidueId(pdb_chain_id, int(pdb_residue_number), pdb_insertion_code)
            is_experimentally_resolved_raw[sequence_identifier].append(True)
            if pdb_residue_name != residue_name:
                # TODO Find a better fix
                residue_name = pdb_residue_name
        sequence_residue = SequenceResidue(residue_name, residue)
        chain_sequences[sequence_identifier].append(sequence_residue)
        polymer_type = entity_poly_type[residue_line["entity_id"]]
        if sequence_identifier in polymer_type_dict and polymer_type_dict[sequence_identifier] != polymer_type:
            raise NPLXV3Error(
                f"Polymer type mismatch for chain {sequence_identifier}: {polymer_type} vs {polymer_type_dict[sequence_identifier]}"
            )
        polymer_type_dict[sequence_identifier] = polymer_type

    is_experimentally_resolved = IsExperimentallyResolvedDict(
        {k: np.array(v) for k, v in is_experimentally_resolved_raw.items()}
    )
    return _SequenceExtractionResult(Sequences(chain_sequences), is_experimentally_resolved, polymer_type_dict)


def _extract_branch_entities(container: PdbxContainers.ContainerBase) -> list[Branch]:
    try:
        pdbx_branch_scheme = extract_cif_data(
            container,
            "pdbx_branch_scheme",
            ["entity_id", "pdb_asym_id", "pdb_seq_num", "pdb_ins_code"],
        )
    except AttributeError:
        return []

    branches = defaultdict(list)
    for residue_line in pdbx_branch_scheme:
        entity_id = residue_line["entity_id"]
        pdb_chain_id = residue_line["pdb_asym_id"]
        pdb_residue_number = residue_line["pdb_seq_num"]
        pdb_ins_code_raw = residue_line["pdb_ins_code"]
        pdb_insertion_code = "" if pdb_ins_code_raw is None else pdb_ins_code_raw.strip("?.")
        if pdb_residue_number.strip() != "?":
            residue = ResidueId(pdb_chain_id, int(pdb_residue_number), pdb_insertion_code)
            branches[(entity_id, pdb_chain_id)].append(residue)

    # TODO check that the branches are actually connected
    return list(branches.values())


def _extract_nonpoly_entities(container: PdbxContainers.ContainerBase, is_crystal: bool) -> list[ResidueId]:
    try:
        pdbx_nonpoly_scheme = extract_cif_data(
            container,
            "pdbx_nonpoly_scheme",
            ["pdb_strand_id", "pdb_seq_num", "pdb_ins_code", "pdb_mon_id"],
        )
    except AttributeError:
        return []

    residues = []
    for i, residue_line in enumerate(pdbx_nonpoly_scheme):
        mon_id = residue_line["pdb_mon_id"]
        if mon_id in SOLVENT_RESNAMES or is_crystal and mon_id in ARTIFACT_RESNAMES:
            continue

        pdb_chain_id = residue_line["pdb_strand_id"]
        pdb_residue_number = residue_line["pdb_seq_num"]
        pdb_ins_code_raw = residue_line["pdb_ins_code"]
        pdb_insertion_code = "" if pdb_ins_code_raw is None else pdb_ins_code_raw.strip("?.")

        if pdb_residue_number.strip() != "?":
            residue = ResidueId(pdb_chain_id, int(pdb_residue_number), pdb_insertion_code)
            residues.append(residue)

    return residues


def _extract_explicit_bonds(container: PdbxContainers.ContainerBase) -> list[ExplicitBond]:
    # TODO requires ptnr1_auth_atom_id instead of ptnr1_label_atom_id, which is not generally available
    # However, ParmED seems to be happy with using ptnr1_label_atom_id, so we'll do the same
    try:
        struct_conn = extract_cif_data(
            container,
            "struct_conn",
            [
                "ptnr1_auth_asym_id",
                "ptnr1_auth_seq_id",
                "pdbx_ptnr1_PDB_ins_code",
                "ptnr1_label_atom_id",
                "ptnr1_symmetry",
                "ptnr2_auth_asym_id",
                "ptnr2_auth_seq_id",
                "pdbx_ptnr2_PDB_ins_code",
                "ptnr2_label_atom_id",
                "ptnr2_symmetry",
                "conn_type_id",
                "pdbx_value_order",
                "pdbx_dist_value",
            ],
        )
    except AttributeError:
        return []
    explicit_bonds: set[ExplicitBond] = set()

    for conn in struct_conn:
        # We'll use an atom distance vs bond length check to ensure that we get the correct atom pairs
        # if conn["ptnr1_symmetry"] != "1_555" or conn["ptnr2_symmetry"] != "1_555":
        #     raise NPLXV3Error("Only 1_555 symmetry is supported")
        # extract bonds
        chain_id_1 = conn["ptnr1_auth_asym_id"]
        chain_id_2 = conn["ptnr2_auth_asym_id"]
        atom1 = AtomIdentifier(
            ResidueId(chain_id_1, int(conn["ptnr1_auth_seq_id"]), conn["pdbx_ptnr1_PDB_ins_code"].strip("?.")),
            conn["ptnr1_label_atom_id"],
        )
        atom2 = AtomIdentifier(
            ResidueId(chain_id_2, int(conn["ptnr2_auth_seq_id"]), conn["pdbx_ptnr2_PDB_ins_code"].strip("?.")),
            conn["ptnr2_label_atom_id"],
        )
        cif_bond_type = (conn["conn_type_id"].strip("?") or None, conn["pdbx_value_order"].strip("?") or None)
        try:
            bond_type = CIF_BOND_TYPE_TO_RDKIT_BOND_TYPE[cif_bond_type]
        except KeyError:
            raise BondError(f"Unsupported bond type: {cif_bond_type}")
        distance_raw = conn["pdbx_dist_value"]
        if distance_raw == "?":
            # For hydrogen bonds, the distance is not always resolved.
            if cif_bond_type[0] != "hydrog":
                raise BondError(f"Missing bond length for bond type {cif_bond_type}")
            distance = 3.3
        else:
            distance = float(conn["pdbx_dist_value"])
        explicit_bonds.add(ExplicitBond(atom1, atom2, int(bond_type), distance))
    return list(explicit_bonds)


def _get_bond_list_from_rdkit_mol(rdkit_mol: Chem.Mol, offset: int = 0) -> list[list[int]]:
    return [
        [bond.GetBeginAtomIdx() + offset, bond.GetEndAtomIdx() + offset, int(bond.GetBondType())]
        for bond in rdkit_mol.GetBonds()  # type: ignore[call-arg, unused-ignore]
    ]

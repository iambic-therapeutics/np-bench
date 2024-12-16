import logging
from collections import defaultdict
from pathlib import Path

import more_itertools
import numpy as np
from parmed import Structure, load_file
from parmed.formats.pdb import _is_hetatm
from rdkit import Chem

from neuralplexer_benchmarks.neuralplexer_data.cif_utils import get_resname_to_type_map
from neuralplexer_benchmarks.neuralplexer_data.constants import ARTIFACT_RESNAMES, ION_RESNAMES, SOLVENT_RESNAMES
from neuralplexer_benchmarks.neuralplexer_data.datamodels import (
    IsExperimentallyResolvedDict,
    Ligand,
    NPLXV3Input,
    PolymerType,
    ResidueId,
    SequenceResidue,
    Sequences,
    validate_nplx_v3_input,
)
from neuralplexer_benchmarks.neuralplexer_data.exceptions import NPLXV3Error
from neuralplexer_benchmarks.neuralplexer_data.smiles_utils import get_smiles_for_resname


def _get_bond_list_from_rdkit_mol(rdkit_mol: Chem.Mol, offset: int = 0) -> list[list[int]]:
    return [
        [bond.GetBeginAtomIdx() + offset, bond.GetEndAtomIdx() + offset, int(bond.GetBondType())]
        for bond in rdkit_mol.GetBonds()  # type: ignore[call-arg, unused-ignore]
    ]


def load_pdb_and_sdfs(
    pdb_path: Path | list[Path] | None,
    sdf_paths: list[Path | Chem.Mol] | None = None,
    *,
    ligand_resnames: list[str] | None = None,
    ligand_smiles: list[str] | None = None,
    parmed_all_residue_template_match: bool = True,
    multiply_b_factor_minus_one: bool = False,
    allow_solvent_entities: bool = False,
    attempt_to_sanitize: bool = True,
) -> NPLXV3Input:
    """
    Load a PDB file and SDF files into a NPLXV3Input object.

    Parameters
    ----------
    pdb_path : Path
        Path to the PDB file. Explicitly set to None if no PDB file is to be loaded.
    sdf_paths : list[Path | Chem.Mol]
        List of paths to the SDF files. Alternatively, rdkit molecules can be passed directly.
    ligand_resnames : list[str], optional
        List of residue names for the ligands. If None, the residue names will be LG1, LG2, etc.
    ligand_smiles : list[str], optional
        List of SMILES strings for the ligands. If None, the SMILES strings will be extracted from the SDF files.
    parmed_all_residue_template_match : bool, optional
        Whether `all_residue_template_match` should be set to True in `parmed.load_file`.
    multiply_b_factor_minus_one : bool, optional
        Whether to multiply the B-factors by -1. This is supposed to be used when the B-factors denoted in the pdb file are actually pLDDT values
    allow_solvent_entities : bool, optional
        Whether solvent residues shall be allowed to be parsed as `nonpolymer_entities`. Note that regardless of this flag, molecules passed in as sdf files are *always* added as `nonpolymer_entities`.
    attempt_to_sanitize : bool, optional
        Whether to attempt to sanitize the rdkit molecules when loading SDF files.
          - If True, will attempt to sanitize the molecule and turn off sanitization if it fails.
          - If False, will not sanitize the molecule.
    """

    if sdf_paths is None:
        sdf_paths = []
    if ligand_resnames is None:
        ligand_resnames = [f"LG{i+1}" for i in range(len(sdf_paths))]
    if len(sdf_paths) != len(ligand_resnames):
        raise ValueError("The number of ligand residue names must match the number of SDF files")
    if ligand_smiles is not None and len(sdf_paths) != len(ligand_smiles):
        raise ValueError("If provided, the number of ligand SMILES strings must match the number of SDF files")
    if isinstance(pdb_path, Path):
        pdb_path = [pdb_path]
    elif pdb_path is None:
        pdb_path = []

    structure = Structure()
    for i, path in enumerate(pdb_path):
        this_structure = load_file(
            str(path),
            expanded_residue_template_match=True,
            all_residue_template_match=parmed_all_residue_template_match,
        )
        if all([not atom.residue.chain for atom in this_structure.atoms]):
            # Set chain to "A" if all atoms have no chain
            for atom in this_structure.atoms:
                atom.residue.chain = chr(ord("A") + i)
        if i == 0:
            structure = this_structure
        else:
            n_bonds_before = len(structure.bonds)
            n_bonds_this_structure = len(this_structure.bonds)
            structure += this_structure
            # Fix bond orders and types, because they are not copied correctly by parmed
            for i in range(n_bonds_this_structure):
                structure.bonds[i + n_bonds_before].order = this_structure.bonds[i].order
                structure.bonds[i + n_bonds_before].qualitative_type = this_structure.bonds[i].qualitative_type

    # All polymer sequences are assumed to be part of the `pdf`, reserving `sdf` files for non-polymer ligands
    chain_sequences: Sequences = defaultdict(list)
    is_experimentally_resolved: dict[str, list[bool]] = defaultdict(list)
    polymer_type: dict[str, PolymerType] = {}
    non_polymer_entities: list[ResidueId] = []
    ligands: list[Ligand] = []

    chain: list[SequenceResidue] = []
    current_sequence_identifier = None
    last_residue_id = None
    residue_types_raw: list[str] = []

    resname_to_raw_type_map = get_resname_to_type_map()

    for atom in structure.atoms:
        chain_id = atom.residue.chain
        this_atom_residue_id = ResidueId(chain_id, atom.residue.number, atom.residue.insertion_code)
        if this_atom_residue_id == last_residue_id:
            continue
        residue_type_raw = resname_to_raw_type_map.get(atom.residue.name, "OTHER")
        is_non_polymer = _raw_types_to_polymer_type([residue_type_raw]) not in [
            PolymerType.DNA,
            PolymerType.RNA,
            PolymerType.L_PEPTIDE,
            PolymerType.D_PEPTIDE,
        ]
        if is_non_polymer:
            if allow_solvent_entities or atom.residue.name not in SOLVENT_RESNAMES:
                non_polymer_entities.append(this_atom_residue_id)
        if chain_id != current_sequence_identifier or is_non_polymer:
            # new chain
            if chain != []:
                if current_sequence_identifier is None:
                    raise NPLXV3Error("Impossible error")  # satisfy mypy
                if not current_sequence_identifier in chain_sequences.keys():
                    # Previously, we raised an error upon duplicate chains, but this caused
                    # issues in edge cases that are potentially very hard to resolve robustly.
                    # Thus, we now pragmatically use the first polymer only.
                    chain_sequences[current_sequence_identifier] = chain
                    is_experimentally_resolved[current_sequence_identifier] = [True] * len(chain)
                    polymer_type[current_sequence_identifier] = _raw_types_to_polymer_type(residue_types_raw)
            current_sequence_identifier = chain_id
            chain = []
            residue_types_raw = []
        last_residue_id = this_atom_residue_id

        if not is_non_polymer:
            residue_types_raw.append(residue_type_raw)
            # Add residue to chain
            chain.append(SequenceResidue(atom.residue.name, this_atom_residue_id))

    # Add last chain as well
    if chain != []:
        if current_sequence_identifier is None:
            raise NPLXV3Error("Impossible error")  # satisfy mypy
        if not current_sequence_identifier in chain_sequences.keys():
            # Previously, we raised an error upon duplicate chains, but this caused
            # issues in edge cases that are potentially very hard to resolve robustly.
            # Thus, we now pragmatically use the first polymer only.
            chain_sequences[current_sequence_identifier] = chain
            is_experimentally_resolved[current_sequence_identifier] = [True] * len(chain)
            polymer_type[current_sequence_identifier] = _raw_types_to_polymer_type(residue_types_raw)

    atomic_numbers_list = [int(atom.atomic_number) for atom in structure.atoms]
    atom_types_pdb_list = [atom.name for atom in structure.atoms]
    atom_coordinates_list = [[atom.xx, atom.xy, atom.xz] for atom in structure.atoms]
    chain_id_list = [atom.residue.chain for atom in structure.atoms]
    residue_id_list = [atom.residue.number for atom in structure.atoms]
    insertion_code_list = [atom.residue.insertion_code.strip("?.") for atom in structure.atoms]
    b_factor_list = [atom.bfactor for atom in structure.atoms]
    if multiply_b_factor_minus_one:
        b_factor_list *= -1
    is_hetatm_list = [_is_hetatm(atom.residue.name) for atom in structure.atoms]

    is_solvent_list = [atom.residue.name in SOLVENT_RESNAMES for atom in structure.atoms]
    is_ion_list = [atom.residue.name in ION_RESNAMES for atom in structure.atoms]

    is_artifact_list = [atom.residue.name in ARTIFACT_RESNAMES for atom in structure.atoms]
    formal_charges_list = [atom.formal_charge or 0 for atom in structure.atoms]

    bond_list = _get_bond_list_from_rdkit_mol(structure.rdkit_mol)

    three_letter_to_smiles = {residue.name: get_smiles_for_resname(residue.name) for residue in structure.residues}

    residue_name_list = [atom.residue.name for atom in structure.atoms]
    offset = len(structure.atoms)
    for i_ligand, (sdf_path, ligand_residue_name) in enumerate(zip(sdf_paths, ligand_resnames)):
        if isinstance(sdf_path, Path):
            rdkit_mol = _load_sdf_to_rdkit_mol(sdf_path, attempt_to_sanitize=attempt_to_sanitize)
        else:
            rdkit_mol = sdf_path

        if rdkit_mol is None:
            raise ValueError(f"Failed to load SDF file: {sdf_path}")

        atomic_symbol_counter: dict[str, int] = defaultdict(lambda: 1)

        ligand_chain_id = f"LG{i_ligand+1}"
        ligand_residue_number = 1
        ligand_insertion_code = ""
        for i, atom in enumerate(rdkit_mol.GetAtoms()):  # type: ignore[call-arg, unused-ignore]
            atomic_numbers_list.append(int(atom.GetAtomicNum()))
            atom_types_pdb_list.append(f"{atom.GetSymbol()}{atomic_symbol_counter[atom.GetSymbol()]}")
            atomic_symbol_counter[atom.GetSymbol()] += 1
            position = rdkit_mol.GetConformer().GetAtomPosition(i)
            atom_coordinates_list.append([position.x, position.y, position.z])
            chain_id_list.append(ligand_chain_id)
            residue_id_list.append(ligand_residue_number)
            insertion_code_list.append(ligand_insertion_code)
            b_factor_list.append(0)
            is_hetatm_list.append(True)
            is_solvent_list.append(False)
            is_ion_list.append(False)
            is_artifact_list.append(False)
            formal_charges_list.append(atom.GetFormalCharge())
            residue_name_list.append(ligand_residue_name)

        bond_list += _get_bond_list_from_rdkit_mol(rdkit_mol, offset=offset)
        offset += rdkit_mol.GetNumAtoms()
        if ligand_smiles is None:
            # We need to remove hydrogens to convert to SMILES;
            # otherwise, we may get "Invariant Violation" error in Chem.MolToSmiles if the SDF file contains hydrogens
            try:
                rdkit_mol_no_hs = Chem.RemoveHs(rdkit_mol)
            except Chem.rdchem.KekulizeException:
                rdkit_mol_no_hs = Chem.RemoveHs(rdkit_mol, sanitize=False)
            three_letter_to_smiles[ligand_residue_name] = Chem.MolToSmiles(rdkit_mol_no_hs)
        else:
            three_letter_to_smiles[ligand_residue_name] = ligand_smiles[i_ligand]
        ligand_residue_id = ResidueId(ligand_chain_id, ligand_residue_number, ligand_insertion_code)
        non_polymer_entities.append(ligand_residue_id)
        ligands.append([ligand_residue_id])

    return validate_nplx_v3_input(
        NPLXV3Input(
            atomic_numbers=np.array(atomic_numbers_list),
            atom_types_pdb=np.array(atom_types_pdb_list),
            atom_coordinates=np.array(atom_coordinates_list),
            chain_id=np.array(chain_id_list),
            is_hetatm=np.array(is_hetatm_list),
            residue_id=np.array(residue_id_list),
            residue_name=np.array(residue_name_list),
            insertion_code=np.array(insertion_code_list),
            b_factor=np.array(b_factor_list),
            bonds=np.array(bond_list),
            chain_sequences=chain_sequences,
            is_experimentally_resolved=IsExperimentallyResolvedDict(
                {k: np.array(v) for k, v in is_experimentally_resolved.items()}
            ),
            polymer_type=polymer_type,
            nonpolymer_entities=non_polymer_entities,
            branch_entities=[],  # can't handle these
            is_solvent=np.array(is_solvent_list),
            is_ion=np.array(is_ion_list),
            is_artifact=np.array(is_artifact_list),
            three_letter_to_smiles=three_letter_to_smiles,
            formal_charges=np.array(formal_charges_list),
            ligands=ligands,
        )
    )


def _raw_types_to_polymer_type(raw_types: list[str]) -> PolymerType:
    RAW_TYPES: set[str] = {
        "D-BETA-PEPTIDE, C-GAMMA LINKING",
        "D-GAMMA-PEPTIDE, C-DELTA LINKING",
        "D-PEPTIDE LINKING",
        "D-PEPTIDE NH3 AMINO TERMINUS",
        "L-BETA-PEPTIDE, C-GAMMA LINKING",
        "L-PEPTIDE COOH CARBOXY TERMINUS",
        "L-PEPTIDE LINKING",
        "L-PEPTIDE NH3 AMINO TERMINUS",
        "L-GAMMA-PEPTIDE, C-DELTA LINKING",
        "PEPTIDE LINKING",
        "PEPTIDE-LIKE",
        "DNA LINKING",
        "DNA OH 3 PRIME TERMINUS",
        "DNA OH 5 PRIME TERMINUS",
        "L-DNA LINKING",
        "RNA LINKING",
        "RNA OH 3 PRIME TERMINUS",
        "RNA OH 5 PRIME TERMINUS",
        "L-RNA LINKING",
        "D-SACCHARIDE",
        "D-SACCHARIDE, ALPHA LINKING",
        "D-SACCHARIDE, BETA LINKING",
        "L-SACCHARIDE",
        "L-SACCHARIDE, ALPHA LINKING",
        "L-SACCHARIDE, BETA LINKING",
        "SACCHARIDE",
        "NON-POLYMER",
        "OTHER",
    }
    if any(raw_type.upper() not in RAW_TYPES for raw_type in raw_types):
        raise ValueError(
            f"Found unknown raw types: {[raw_type.upper() for raw_type in raw_types if raw_type.upper() not in RAW_TYPES]}"
        )
    is_l_peptide = lambda x: "PEPTIDE" in x.upper() and not x.upper().startswith("D-")
    is_d_peptide = lambda x: "PEPTIDE" in x.upper() and not x.upper().startswith("L-")
    is_dna = lambda x: "DNA " in x.upper()
    is_rna = lambda x: "RNA " in x.upper()
    is_l_saccharide = lambda x: "SACCHARIDE" in x.upper() and not x.upper().startswith("D-")
    is_d_saccharide = lambda x: "SACCHARIDE" in x.upper() and not x.upper().startswith("L-")

    if all(is_l_peptide(raw_type) for raw_type in raw_types):
        return PolymerType.L_PEPTIDE
    if all(is_d_peptide(raw_type) for raw_type in raw_types):
        return PolymerType.D_PEPTIDE
    if all(is_dna(raw_type) for raw_type in raw_types):
        return PolymerType.DNA
    if all(is_rna(raw_type) for raw_type in raw_types):
        return PolymerType.RNA
    if all(is_l_saccharide(raw_type) for raw_type in raw_types):
        return PolymerType.L_SACCHARIDE
    if all(is_d_saccharide(raw_type) for raw_type in raw_types):
        return PolymerType.D_SACCHARIDE
    return PolymerType.OTHER


def _load_sdf_to_rdkit_mol(sdf_path: Path, attempt_to_sanitize: bool = True) -> Chem.Mol:
    rdkit_mols = Chem.SDMolSupplier(str(sdf_path), sanitize=attempt_to_sanitize)

    if len(rdkit_mols) > 1:
        raise ValueError(f"Expected a single structure in SDF file: {sdf_path}")

    rdkit_mol = more_itertools.one(rdkit_mols)
    if rdkit_mol is None:
        logging.warning(f"Sanitization failed when loading SDF to RDKit Mol. Retrying with sanitize=False")
        rdkit_mol = more_itertools.one(Chem.SDMolSupplier(str(sdf_path), sanitize=False))
        if rdkit_mol is None:
            raise ValueError(f"Failed to load SDF file: {sdf_path}")

    return rdkit_mol

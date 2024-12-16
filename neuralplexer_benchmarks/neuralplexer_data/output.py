import logging
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Literal

import numpy as np
from more_itertools import first
from parmed import Structure
from parmed.topologyobjects import Atom, Bond
from rdkit import Chem
from rdkit.Chem import KekulizeException

from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input, ResidueId, SequenceResidue
from neuralplexer_benchmarks.str_enum import StrEnum


class SplittingStrategy(StrEnum):
    CONNECTIVITY = "connectivity"
    RESIDUE_NAME = "residue_name"
    NO_SPLIT = "no_split"


@dataclass
class LigandStrategy:
    residue_names: list[str] | list[ResidueId] | None | Callable[[str], bool] = (
        None  # If a list is provided, atoms with these residue names (or ResidueIds) will be considered ligands; if it is None, all HETATM atoms will be considered ligands; if a function is provided, each residue name is tested with this function in order to decide whether it is a ligand
    )
    splitting_strategy: SplittingStrategy = SplittingStrategy.CONNECTIVITY


def dump_nplx_v3_input_to_pdb_and_sdf(
    nplx_v3_input: NPLXV3Input,
    ligand_strategy: LigandStrategy = LigandStrategy(),
    pdb_strategy: list[str] | Literal["all"] | Literal["near_ligand"] = "all",
    attempt_to_kekulize: bool = True,
) -> tuple[str, list[str]]:
    """
    pdb_strategy:
    - "all": all non-ligand (non-solvent, non-artifact, non-ion) atoms are considered receptor atoms
    - "near_ligand": all non-ligand atoms that are within 5 Å of a ligand atom are considered receptor atoms
    - list of sequence IDs: all atoms in the specified polymer sequences are considered receptor atoms
    """
    ligand_atoms, possible_receptor_atoms = _get_ligand_and_receptor_atoms(
        nplx_v3_input, residue_names=ligand_strategy.residue_names
    )
    ligand_atom_bonds = nplx_v3_input.bonds[
        np.logical_and(
            ligand_atoms[nplx_v3_input.bonds[:, 0]],
            ligand_atoms[nplx_v3_input.bonds[:, 1]],
        )
    ]

    splitting_strategy = ligand_strategy.splitting_strategy
    if splitting_strategy == SplittingStrategy.NO_SPLIT:
        ligands_atom_indices = [set(np.where(ligand_atoms)[0])]
    elif splitting_strategy == SplittingStrategy.RESIDUE_NAME:
        ligands_atom_indices = []
        for residue_name in np.unique(nplx_v3_input.residue_name[ligand_atoms]):
            ligands_atom_indices.append(set(np.where(nplx_v3_input.residue_name == residue_name)[0]))
    elif splitting_strategy == SplittingStrategy.CONNECTIVITY:
        ligands_atom_indices = _split_ligands_by_connectivity(ligand_atoms, ligand_atom_bonds)
    else:
        raise ValueError("Invalid splitting strategy")

    # Create SDF block for each ligand
    ligand_sdf_blocks = []
    for this_ligand_atom_indices in ligands_atom_indices:
        ligand_sdf_blocks.append(
            _create_sdf_block(sorted(this_ligand_atom_indices), ligand_atom_bonds, nplx_v3_input, attempt_to_kekulize)
        )

    residue_identifier_to_sequence_id = {}
    for sequence_id, sequence in nplx_v3_input.chain_sequences.items():
        for residue in sequence:
            if residue.residue_id:
                residue_identifier_to_sequence_id[residue.residue_id] = sequence_id

    if pdb_strategy == "all":
        receptor_atoms = possible_receptor_atoms
    elif pdb_strategy == "near_ligand":
        # Find all atoms that are not ligands but are within 5 Å of a ligand
        receptor_atoms = possible_receptor_atoms
        for ligand_atom_indices in ligands_atom_indices:
            for ligand_atom_index in ligand_atom_indices:
                ligand_atom_coordinates = nplx_v3_input.atom_coordinates[ligand_atom_index]
                for receptor_atom_index in np.where(possible_receptor_atoms)[0]:
                    receptor_atom_coordinates = nplx_v3_input.atom_coordinates[receptor_atom_index]
                    if np.linalg.norm(ligand_atom_coordinates - receptor_atom_coordinates) > 5:
                        receptor_atoms[receptor_atom_index] = False
    elif isinstance(pdb_strategy, list):
        residue_ids_per_atom = [
            ResidueId(nplx_v3_input.chain_id[i], nplx_v3_input.residue_id[i], nplx_v3_input.insertion_code[i])
            for i in range(len(nplx_v3_input.atomic_numbers))
        ]
        receptor_residues: list[SequenceResidue] = sum(
            (
                sequence
                for sequence_id, sequence in nplx_v3_input.chain_sequences.items()
                if sequence_id in pdb_strategy
            ),
            [],
        )
        receptor_residue_ids = [residue.residue_id for residue in receptor_residues]
        receptor_atoms = (
            np.array([residue_id in receptor_residue_ids for residue_id in residue_ids_per_atom])
            & possible_receptor_atoms
        )
    else:
        raise ValueError("Invalid pdb_strategy")
    # Create ParmED structure for the receptor
    receptor_structure = _create_receptor_structure(receptor_atoms, nplx_v3_input)
    pdb_string_io = StringIO()
    receptor_structure.write_pdb(pdb_string_io, renumber=False)

    return pdb_string_io.getvalue(), ligand_sdf_blocks


def _get_ligand_and_receptor_atoms(
    nplx_v3_input: NPLXV3Input, residue_names: list[str] | list[ResidueId] | None | Callable[[str], bool] = None
) -> tuple[np.ndarray, np.ndarray]:
    ligand_and_receptor_atoms = ~(nplx_v3_input.is_ion | nplx_v3_input.is_solvent | nplx_v3_input.is_artifact)
    if isinstance(residue_names, list):
        if all(isinstance(residue_name, str) for residue_name in residue_names):
            ligand_atoms = np.array([residue_name in residue_names for residue_name in nplx_v3_input.residue_name])
        elif all(isinstance(residue_name, ResidueId) for residue_name in residue_names):
            residue_ids_per_atom = [
                ResidueId(nplx_v3_input.chain_id[i], nplx_v3_input.residue_id[i], nplx_v3_input.insertion_code[i])
                for i in range(len(nplx_v3_input.atomic_numbers))
            ]
            ligand_atoms = np.array([residue_id in residue_names for residue_id in residue_ids_per_atom])
        else:
            raise ValueError("All elements in the list must be either strings or ResidueIds")
        receptor_atoms = ~ligand_atoms & ligand_and_receptor_atoms
    elif residue_names is None:
        ligand_atoms = nplx_v3_input.is_hetatm & ligand_and_receptor_atoms
        receptor_atoms = ~nplx_v3_input.is_hetatm & ligand_and_receptor_atoms
    else:
        ligand_atoms = np.array([residue_names(residue_name) for residue_name in nplx_v3_input.residue_name])
        receptor_atoms = ~ligand_atoms & ligand_and_receptor_atoms
    return ligand_atoms, receptor_atoms


def _get_adjacency_list(ligand_atom_bonds: np.ndarray) -> dict[int, set[int]]:
    adjacency_list: dict[int, set[int]] = defaultdict(set)
    for i, j, _ in ligand_atom_bonds:
        adjacency_list[i].add(j)
        adjacency_list[j].add(i)
    return adjacency_list


def _split_ligands_by_connectivity(ligand_atoms: np.ndarray, ligand_atom_bonds: np.ndarray) -> list[set[int]]:
    adjacency_list = _get_adjacency_list(ligand_atom_bonds)

    ligands_atom_indices: list[set[int]] = []
    visited_global: set[int] = set()

    # Use BFS algorithm to find the connected components
    def bfs(node: int) -> set[int]:
        visited = set()
        queue = [node]
        visited.add(node)
        while queue:
            node = queue.pop(0)
            for neighbour in adjacency_list.get(node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return visited

    any_ligand_atom_indices = np.where(ligand_atoms)[0]

    while True:
        node = first(filter(lambda x: x not in visited_global, any_ligand_atom_indices), None)
        if node is None:
            break
        visited = bfs(node)
        visited_global.update(visited)
        ligands_atom_indices.append(visited)

    return ligands_atom_indices


NPLXV3_LIGAND_SDF_HEADER = "Created by Neuralplexer v3\n"


def _create_sdf_block(
    this_ligand_atom_indices: list[int],
    ligand_atom_bonds: np.ndarray,
    nplx_v3_input: NPLXV3Input,
    attempt_to_kekulize: bool = True,
) -> str:
    indices_this_rdkit_mol = []
    # Create an rdkit molecule first, using the bond information from the input
    ligand_molecule = Chem.RWMol()
    for atom_index_global in this_ligand_atom_indices:
        atom = Chem.Atom(int(nplx_v3_input.atomic_numbers[atom_index_global]))
        if nplx_v3_input.formal_charges is not None:
            atom.SetFormalCharge(int(nplx_v3_input.formal_charges[atom_index_global]))
        ligand_molecule.AddAtom(atom)
        indices_this_rdkit_mol.append(atom_index_global)
    # Add coordinates for a conformer
    conformer = Chem.Conformer(len(this_ligand_atom_indices))
    for atom_index_in_mol, atom_index_global in enumerate(this_ligand_atom_indices):
        conformer.SetAtomPosition(atom_index_in_mol, nplx_v3_input.atom_coordinates[atom_index_global])
    ligand_molecule.AddConformer(conformer, assignId=True)
    for i_global, j_global, bond_type in ligand_atom_bonds:
        if bond_type == 99:
            bond_type = int(Chem.rdchem.BondType.SINGLE)
        if i_global in this_ligand_atom_indices and j_global in this_ligand_atom_indices:
            i_mol = indices_this_rdkit_mol.index(i_global)
            j_mol = indices_this_rdkit_mol.index(j_global)
            ligand_molecule.AddBond(i_mol, j_mol, Chem.rdchem.BondType(bond_type))
    # Convert the rdkit molecule to an sdf block
    ligand_mol = ligand_molecule.GetMol()
    sdf_block = _rdkit_mol_to_sdf_block(ligand_mol, attempt_to_kekulize)
    return sdf_block


def _rdkit_mol_to_sdf_block(mol: Chem.Mol | None, attempt_to_kekulize: bool = True) -> str:
    if mol is None:
        raise ValueError("Could not convert a NoneType RDKit Mol to an SDF block")
    try:
        mol_block = Chem.MolToMolBlock(mol, kekulize=attempt_to_kekulize)
    except KekulizeException:
        logging.warning("Can't kekulize molecule, trying without kekulization")
        mol_block = Chem.MolToMolBlock(mol, kekulize=False)

    return NPLXV3_LIGAND_SDF_HEADER.rstrip("\n") + mol_block


def _create_receptor_structure(receptor_atoms: np.ndarray, nplx_v3_input: NPLXV3Input) -> Structure:
    residue_identifier_to_sequence_id = {}
    for sequence_id, sequence in nplx_v3_input.chain_sequences.items():
        for residue in sequence:
            if residue.residue_id:
                residue_identifier_to_sequence_id[residue.residue_id] = sequence_id

    receptor_structure = Structure()
    indices_receptor = np.where(receptor_atoms)[0]
    for i in indices_receptor:
        x, y, z = nplx_v3_input.atom_coordinates[i]
        residue_id = ResidueId(nplx_v3_input.chain_id[i], nplx_v3_input.residue_id[i], nplx_v3_input.insertion_code[i])
        seg_id = residue_identifier_to_sequence_id.get(residue_id, "")
        formal_charge = nplx_v3_input.formal_charges[i] if nplx_v3_input.formal_charges is not None else None
        atom = Atom(
            name=nplx_v3_input.atom_types_pdb[i],
            atomic_number=nplx_v3_input.atomic_numbers[i],
            number=i + 1,
            occupancy=1.0,
            bfactor=nplx_v3_input.b_factor[i],
            formal_charge=formal_charge,
        )
        atom.xx, atom.xy, atom.xz = x, y, z
        receptor_structure.add_atom(
            atom,
            nplx_v3_input.residue_name[i],
            nplx_v3_input.residue_id[i],
            nplx_v3_input.chain_id[i],
            nplx_v3_input.insertion_code[i],
            seg_id,
        )
    receptor_structure.positions = nplx_v3_input.atom_coordinates[receptor_atoms]
    indices_receptor_inverse = {index: i for i, index in enumerate(indices_receptor)}
    # Note: ParmED does not write CONECT records in its pdb output. The bonding
    # information is only relevant for ParmED to correctly terminate polymer chains.
    # The actual bond type is ignored.
    for bond in nplx_v3_input.bonds[
        np.logical_and(
            receptor_atoms[nplx_v3_input.bonds[:, 0]],
            receptor_atoms[nplx_v3_input.bonds[:, 1]],
        )
    ]:
        if bond[2] == 99:
            bond[2] = int(Chem.rdchem.BondType.SINGLE)
        receptor_structure.bonds.append(
            Bond(
                receptor_structure.atoms[indices_receptor_inverse[bond[0]]],
                receptor_structure.atoms[indices_receptor_inverse[bond[1]]],
                qualitative_type=bond[2] if bond[2] != -1 else 0,
            )
        )
    return receptor_structure

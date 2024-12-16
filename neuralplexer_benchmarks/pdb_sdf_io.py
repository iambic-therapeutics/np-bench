from pathlib import Path

import more_itertools
import numpy as np
from rdkit import Chem

from neuralplexer_benchmarks.neuralplexer_data.augmentation import drop_all_hydrogen_atoms, drop_solvent
from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input
from neuralplexer_benchmarks.neuralplexer_data.output import (
    LigandStrategy,
    _get_ligand_and_receptor_atoms,
    _split_ligands_by_connectivity,
    dump_nplx_v3_input_to_pdb_and_sdf,
)
from neuralplexer_benchmarks.neuralplexer_data.pdb_sdf_io import load_pdb_and_sdfs


def load_rdkit_mol(sdf_path: Path, attempt_to_sanitize: bool = True) -> tuple[Chem.Mol, bool]:
    """Load a molecule from an sdf/mol file.

    If attempt_to_sanitize is True, will try to load the molecule with sanitization, if it fails, try without sanitization.

    Returns:
        tuple[Chem.Mol, bool]: RDKit Mol object and a boolean indicating if sanitization was successful.
    """
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=attempt_to_sanitize)
    if mol is not None:
        return mol, attempt_to_sanitize
    else:
        mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False)
        if mol is None:
            raise ValueError(f"Failed to load sdf/mol file '{sdf_path}' into RDKit Mol object.")
        else:
            return mol, False


def make_nplx_v3_input(
    protein_pdb: Path, ligand_sdfs: list[Path] | None = None, parmed_all_residue_template_match: bool = False
) -> NPLXV3Input:
    """Make an NPLXv3 input from the protein PDB and ligand SDF files."""
    nplx_v3_input = load_pdb_and_sdfs(
        protein_pdb, ligand_sdfs, parmed_all_residue_template_match=parmed_all_residue_template_match
    )
    nplx_v3_input = drop_all_hydrogen_atoms(nplx_v3_input, prob=1.0)
    nplx_v3_input = drop_solvent(nplx_v3_input, prob=1.0)
    return nplx_v3_input


def rewrite_pdb_with_nplx_v3_input(
    protein_pdb: Path,
    output_protein_pdb: Path,
) -> None:
    """Rewrite the PDB file with the NPLXV3Input format.

    The input pdb file is loaded to an NPLXV3Input object, then the object is dumped to a new pdb file.

    This is needed to ensure that the PDB file is in the correct format for metrics calculation.
    """
    nplx_v3_input = make_nplx_v3_input(
        protein_pdb=protein_pdb, ligand_sdfs=None, parmed_all_residue_template_match=False
    )

    ligand_strategy = LigandStrategy(residue_names=lambda x: x.startswith("LG"))

    pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(nplx_v3_input, ligand_strategy=ligand_strategy)

    output_protein_pdb.write_text(pdb_str)


def rewrite_ligand_sdf_with_reference_sdf(
    pred_ligand_sdf: Path,
    ref_ligand_sdf: Path,
    output_ligand_sdf: Path,
    match_atom_ordering: bool = True,
    attempt_to_sanitize: bool = True,
) -> None:
    """
    Rewrite the ligand SDF file with the reference ligand SDF file as a template.

    The output ligand SDF file will have the same atom coordinates as the input ligand SDF file,
    but all the other information will be taken from the reference ligand SDF file.

    If match_atom_ordering is True, the atom ordering in the input ligand SDF file will be matched with the reference;
    otherwise, an error will be raised if the atom ordering is different.
    """
    # First load the predicted SDF and return the sanitization status
    this_mol, sanitize = load_rdkit_mol(pred_ligand_sdf, attempt_to_sanitize=attempt_to_sanitize)

    # Load the reference SDF with the same sanitization status as the predicted SDF
    ref_mol, _ = load_rdkit_mol(ref_ligand_sdf, attempt_to_sanitize=sanitize)

    if this_mol.GetNumAtoms() != ref_mol.GetNumAtoms():
        raise ValueError("The pred ligand SDF has different number of atoms than the ref ligand SDF.")

    ref_atom_types = [atom.GetSymbol() for atom in ref_mol.GetAtoms()]  # type: ignore[call-arg, unused-ignore]
    this_atom_types = [atom.GetSymbol() for atom in this_mol.GetAtoms()]  # type: ignore[call-arg, unused-ignore]

    if ref_atom_types != this_atom_types:
        match_indices = list(this_mol.GetSubstructMatch(ref_mol))
        if len(match_indices) == 0:
            raise ValueError("Failed to match the atoms in the pred and ref ligand SDFs.")
        if match_indices != list(range(this_mol.GetNumAtoms())):
            if match_atom_ordering:
                this_mol = Chem.RenumberAtoms(this_mol, match_indices)
            else:
                raise ValueError("The pred ligand SDF has different atom ordering than the ref ligand SDF.")

    conformer = ref_mol.GetConformer()

    xyz_predicted = np.array(this_mol.GetConformer().GetPositions())
    for atom_index_in_mol in range(len(xyz_predicted)):
        conformer.SetAtomPosition(atom_index_in_mol, xyz_predicted[atom_index_in_mol])

    output_sdf_block = Chem.MolToMolBlock(ref_mol)
    output_ligand_sdf.write_text(output_sdf_block)


def get_chain_ids_for_ligand_residue_name(nplx_v3_input: NPLXV3Input, ligand_residue_name: str) -> list[str]:
    """
    Get the chain IDs for the ligand residue name.

    The chain IDs are ordered by the order of the ligands in the NPLXV3Input.
    """
    ligand_atoms, _ = _get_ligand_and_receptor_atoms(nplx_v3_input, residue_names=[ligand_residue_name])

    ligand_atom_bonds = nplx_v3_input.bonds[
        np.logical_and(
            ligand_atoms[nplx_v3_input.bonds[:, 0]],
            ligand_atoms[nplx_v3_input.bonds[:, 1]],
        )
    ]

    ligands_atom_indices = _split_ligands_by_connectivity(ligand_atoms, ligand_atom_bonds)

    def _get_this_ligand_chain_id(this_ligand_atom_indices: set[int]) -> str:
        this_ligand_chain_ids: set[str] = set()
        for atom_index_global in this_ligand_atom_indices:
            this_ligand_chain_ids.add(nplx_v3_input.chain_id[atom_index_global])
        if len(this_ligand_chain_ids) != 1:
            raise ValueError(f"Expected exactly one chain ID for this ligand, got {this_ligand_chain_ids}")
        chain_id = more_itertools.one(this_ligand_chain_ids)
        return chain_id

    chain_ids = []
    for this_ligand_atom_indices in ligands_atom_indices:
        chain_id = _get_this_ligand_chain_id(this_ligand_atom_indices)
        chain_ids.append(chain_id)

    return chain_ids

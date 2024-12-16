from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from neuralplexer_benchmarks.pdb_sdf_io import load_rdkit_mol


def _get_reference_poses(sdf_file: Path, sanitize: bool = True) -> list[Chem.Mol]:
    poses = []
    suppl = Chem.SDMolSupplier(sdf_file.as_posix(), sanitize=sanitize)
    for mol in suppl:
        if mol is not None:
            poses.append(mol)

    if not poses:
        raise ValueError("Failed to load reference poses.")

    return poses


def compute_ligand_symrmsd(
    ref_mol_file: Path,
    pred_mol_file: Path,
    attempt_to_sanitize: bool = True,
) -> float:
    """Compute the symmetric RMSD between two molecules.
    Args:
        ref_mol_file (Path): Path to the reference molecule file.
        aligned_mol_file (Path): Path to the aligned molecule file.
        attempt_to_sanitize (bool): Attempt to sanitize the molecules before computing the RMSD.

    Returns:
        float: Symmetric RMSD value.
    """
    docked_mol, sanitize = load_rdkit_mol(pred_mol_file, attempt_to_sanitize=attempt_to_sanitize)
    n_atoms_docked = docked_mol.GetNumAtoms()

    poses = _get_reference_poses(ref_mol_file, sanitize=sanitize)
    rmsds = []
    for pose in poses:
        if (n_atoms_ref := pose.GetNumAtoms()) != n_atoms_docked:
            raise ValueError(
                f"Number of atoms mismatch between aligned ligand ({n_atoms_docked}) and reference ligand ({n_atoms_ref})."
            )

        rms = rdMolAlign.CalcRMS(docked_mol, pose)
        rmsds.append(rms)

    return min(rmsds)

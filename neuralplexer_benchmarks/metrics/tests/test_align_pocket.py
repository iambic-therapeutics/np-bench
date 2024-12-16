import pytest

from neuralplexer_benchmarks.metrics.align_pocket import compute_best_pocket_aligned_rmsd


def test_compute_best_pocket_aligned_rmsd(
    protein_pdb_path,
    ligand_sdf_path,
    predicted_protein_pdb_path,
    predicted_ligand_sdf_path,
    tmp_path,
):
    aligned_ligand_file = tmp_path / "aligned_ligand.sdf"

    ligand_rmsd, pocket_rmsd = compute_best_pocket_aligned_rmsd(
        ref_protein_pdb=protein_pdb_path,
        ref_ligand_sdf=ligand_sdf_path,
        pred_protein_pdb=predicted_protein_pdb_path,
        pred_ligand_sdf=predicted_ligand_sdf_path,
        aligned_ligand_sdf=aligned_ligand_file,
        work_dir=tmp_path,
    )

    # Reference RMSD from running the python API of PyMOL
    reference_pocket_rmsd = 3.342
    reference_ligand_rmsd = 4.911

    assert pocket_rmsd == pytest.approx(reference_pocket_rmsd, abs=1e-3)
    assert ligand_rmsd == pytest.approx(reference_ligand_rmsd, abs=1e-3)

    assert aligned_ligand_file.is_file()

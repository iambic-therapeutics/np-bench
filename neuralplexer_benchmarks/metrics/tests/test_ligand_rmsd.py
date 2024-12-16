from neuralplexer_benchmarks.metrics.ligand_rmsd import compute_ligand_symrmsd


def test_compute_ligand_symrmsd(ligand_sdf_path, aligned_ligand_sdf_path):
    symrmsd = compute_ligand_symrmsd(
        ref_mol_file=ligand_sdf_path,
        pred_mol_file=aligned_ligand_sdf_path,
    )

    assert symrmsd < 10

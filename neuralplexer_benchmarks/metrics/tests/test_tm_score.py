from neuralplexer_benchmarks.metrics.tm_score import compute_protein_tm_score_and_rmsd


def test_compute_protein_tm_score_and_rmsd(
    protein_pdb_path,
    predicted_protein_pdb_path,
):
    tm_score, rmsd = compute_protein_tm_score_and_rmsd(
        pdb_file=predicted_protein_pdb_path,
        ref_pdb_file=protein_pdb_path,
    )

    assert tm_score < 1
    assert rmsd < 42

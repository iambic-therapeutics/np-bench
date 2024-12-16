from pathlib import Path

import pytest

from neuralplexer_benchmarks.internal_evaluation.generalized_rmsd import compute_generalized_ligand_rmsds
from neuralplexer_benchmarks.neuralplexer_data.datamodels import load_nplx_v3_input


def test_compute_generalized_ligand_rmsds(ref_multi_ligand_npz: Path, pred_multi_ligand_npz: Path) -> None:
    reference_nplx_v3_input = load_nplx_v3_input(ref_multi_ligand_npz)
    predicted_nplx_v3_input = load_nplx_v3_input(pred_multi_ligand_npz)

    ligand_polymer_interfaces = {("ACO", "A"): "lig:ACO||poly:A", ("ACO", "B"): "lig:ACO||poly:B"}

    results = compute_generalized_ligand_rmsds(
        reference_nplx_v3_input, predicted_nplx_v3_input, ligand_polymer_interfaces
    )

    assert len(results) == 2
    assert set(results.keys()) == {"lig:ACO||poly:A", "lig:ACO||poly:B"}

    reference_rmsds = [4.109672678965478, 4.184540392992786]

    assert results["lig:ACO||poly:A"] == pytest.approx(reference_rmsds[0], abs=1e-3)
    assert results["lig:ACO||poly:B"] == pytest.approx(reference_rmsds[1], abs=1e-3)

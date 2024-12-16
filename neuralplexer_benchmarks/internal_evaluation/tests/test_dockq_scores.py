import pytest

from neuralplexer_benchmarks.internal_evaluation.dockq_scores import compute_dockq_scores
from neuralplexer_benchmarks.neuralplexer_data.datamodels import load_nplx_v3_input


def test_compute_dockq_scores(ref_multi_chain_npz, pred_multi_chain_npz):
    reference_nplx_v3_input = load_nplx_v3_input(ref_multi_chain_npz)
    predicted_nplx_v3_input = load_nplx_v3_input(pred_multi_chain_npz)

    interface_map = {("A", "B"): "poly:A||poly:B"}

    # check that the DockQ scores are computed correctly
    dockq_scores, best_mapping = compute_dockq_scores(reference_nplx_v3_input, predicted_nplx_v3_input, interface_map)

    reference_dockq_score = 0.446

    assert len(dockq_scores) == 1
    assert set(dockq_scores.keys()) == {"poly:A||poly:B"}
    assert dockq_scores["poly:A||poly:B"] == pytest.approx(reference_dockq_score, abs=1e-3)

    assert best_mapping == {"A": "A", "B": "B"}

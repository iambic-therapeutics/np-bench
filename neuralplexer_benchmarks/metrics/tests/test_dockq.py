import pytest

from neuralplexer_benchmarks.metrics.dockq import (
    DockQResult,
    _parse_dockq_cli_output,
    run_dockq_cli,
    run_dockq_python_api,
)


@pytest.fixture(name="reference_dockq_score")
def fixture_reference_dockq_score():
    # Reference DockQ score between ref_multi_chain_pdb, pred_multi_chain_pdb
    return 0.446


def test_run_dockq_cli(protein_pdb_path, ref_multi_chain_pdb, pred_multi_chain_pdb, reference_dockq_score):
    # check that the DockQ CLI returns None for a single chain protein
    dockq_result = run_dockq_cli(native_pdb_path=protein_pdb_path, model_pdb_path=protein_pdb_path)
    assert dockq_result is None

    # check that the DockQ CLI returns a DockQResult for a multi-chain protein
    dockq_result = run_dockq_cli(native_pdb_path=ref_multi_chain_pdb, model_pdb_path=pred_multi_chain_pdb)
    assert dockq_result is not None
    assert isinstance(dockq_result, DockQResult)

    assert dockq_result.total_dockq == pytest.approx(reference_dockq_score, abs=1e-3)
    assert dockq_result.best_chain_mapping == {"A": "A", "B": "B"}
    assert len(dockq_result.interface_results) == 1
    assert set(dockq_result.interface_results.keys()) == {("A", "B")}
    assert dockq_result.interface_results[("A", "B")]["DockQ"] == pytest.approx(reference_dockq_score, abs=1e-3)


def test_run_dockq_python_api(protein_pdb_path, ref_multi_chain_pdb, pred_multi_chain_pdb, reference_dockq_score):
    # check that the DockQ Python API returns empty dict for a single chain protein
    result = run_dockq_python_api(native_pdb_path=protein_pdb_path, model_pdb_path=protein_pdb_path)
    assert len(result) == 0

    # check that the DockQ Python API returns a non-empty dict for a multi-chain protein
    result = run_dockq_python_api(native_pdb_path=ref_multi_chain_pdb, model_pdb_path=pred_multi_chain_pdb)

    assert result["model"] == str(pred_multi_chain_pdb)
    assert result["native"] == str(ref_multi_chain_pdb)
    assert result["best_dockq"] == pytest.approx(reference_dockq_score, abs=1e-3)
    assert result["best_mapping"] == {"A": "A", "B": "B"}
    assert result["best_mapping_str"] == "AB:AB"

    best_result = result["best_result"]
    assert len(best_result) == 1
    assert set(best_result.keys()) == {"AB"}

    best_result_chain_map = best_result["AB"]
    assert best_result_chain_map["DockQ"] == pytest.approx(reference_dockq_score, abs=1e-3)
    assert best_result_chain_map["chain_map"] == {"A": "A", "B": "B"}


def test_parse_dockq_cli_output():
    # Sample output with multiple blocks of native pairs
    output = """
Total DockQ over 3 native interfaces: 0.653 with BAC:ABC model:native mapping
Native chains: A, B
	Model chains: B, A
	DockQ: 0.994
	irms: 0.000
	Lrms: 0.000
	fnat: 0.983
	fnonnat: 0.008
	clashes: 0.000
	F1: 0.987
	DockQ_F1: 0.996
Native chains: A, C
	Model chains: B, C
	DockQ: 0.511
	irms: 1.237
	Lrms: 6.864
	fnat: 0.333
	fnonnat: 0.000
	clashes: 0.000
	F1: 0.500
	DockQ_F1: 0.567
Native chains: B, C
	Model chains: A, C
	DockQ: 0.453
	irms: 2.104
	Lrms: 8.131
	fnat: 0.500
	fnonnat: 0.107
	clashes: 0.000
	F1: 0.641
	DockQ_F1: 0.500
"""

    dockq_result = _parse_dockq_cli_output(output)

    assert dockq_result is not None
    assert isinstance(dockq_result, DockQResult)

    interface_results = dockq_result.interface_results
    best_mapping = dockq_result.best_chain_mapping

    assert len(interface_results) == 3
    assert best_mapping == {"A": "B", "B": "A", "C": "C"}

    assert interface_results[("A", "B")]["DockQ"] == 0.994
    assert interface_results[("A", "C")]["DockQ"] == 0.511
    assert interface_results[("B", "C")]["DockQ"] == 0.453

    # check model chains
    assert interface_results[("A", "B")]["Model chains"] == ("B", "A")
    assert interface_results[("A", "C")]["Model chains"] == ("B", "C")
    assert interface_results[("B", "C")]["Model chains"] == ("A", "C")

    # check Lrms
    assert interface_results[("A", "B")]["Lrms"] == 0.000
    assert interface_results[("A", "C")]["Lrms"] == 6.864
    assert interface_results[("B", "C")]["Lrms"] == 8.131

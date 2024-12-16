from pathlib import Path

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from neuralplexer_benchmarks.internal_evaluation.internal_metrics import compute_internal_metrics


def test_compute_internal_metrics_successful(
    ref_multi_chain_npz: Path,
    pred_multi_chain_npz: Path,
) -> None:
    metrics_df, failures_df = compute_internal_metrics(
        mmcif_id="test_id",
        chain_or_interface_ids=["poly:A||poly:B", "poly:A"],
        ref_npz_path=ref_multi_chain_npz,
        pred_npz_path=pred_multi_chain_npz,
    )

    assert isinstance(metrics_df, pd.DataFrame)
    assert len(failures_df) == 0

    assert metrics_df["mmcif_id"].to_list() == ["test_id"] * len(metrics_df)
    assert metrics_df["chain_or_interface_id"].to_list() == [
        "poly:A||poly:B",
        "global",
        "poly:A",
    ]
    assert metrics_df["metrics"].to_list() == [
        "DockQ",
        "tm_score",
        "tm_score",
    ]


def test_compute_internal_metrics_with_errors(
    ref_multi_chain_npz: Path,
    pred_multi_chain_npz: Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "neuralplexer_benchmarks.internal_evaluation.internal_metrics.compute_dockq_scores",
        side_effect=Exception("dockq error"),
    )
    mocker.patch(
        "neuralplexer_benchmarks.internal_evaluation.internal_metrics.compute_tm_scores",
        side_effect=Exception("tm score error"),
    )
    mocker.patch(
        "neuralplexer_benchmarks.internal_evaluation.internal_metrics.compute_generalized_ligand_rmsds",
        side_effect=Exception("rmsd error"),
    )

    # Check that exception is raised when return_exceptions is False
    with pytest.raises(ValueError, match="Exceptions occurred during metric computation"):
        compute_internal_metrics(
            mmcif_id="test_id",
            chain_or_interface_ids=["poly:A||poly:B", "lig:ABC||poly:A", "poly:A"],
            ref_npz_path=ref_multi_chain_npz,
            pred_npz_path=pred_multi_chain_npz,
            return_exceptions=False,
        )

    # Check that exceptions are returned when return_exceptions is True
    metrics_df, failures_df = compute_internal_metrics(
        mmcif_id="test_id",
        chain_or_interface_ids=["poly:A||poly:B", "lig:ABC||poly:A", "poly:A"],
        ref_npz_path=ref_multi_chain_npz,
        pred_npz_path=pred_multi_chain_npz,
        return_exceptions=True,
    )

    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 0

    assert len(failures_df) == 3
    assert failures_df["error_msg"][0] == "DockQError: dockq error"
    assert failures_df["error_msg"][1] == "GeneralizedRMSDError: rmsd error"
    assert failures_df["error_msg"][2] == "TMScoreError: tm score error"

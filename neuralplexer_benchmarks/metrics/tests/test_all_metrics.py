from pathlib import Path

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from neuralplexer_benchmarks.metrics.all_metrics import compute_metrics


def test_compute_metrics_successful(
    protein_pdb_path: Path,
    ligand_sdf_path: Path,
    predicted_protein_pdb_path: Path,
    predicted_ligand_sdf_path: Path,
    tmp_path: Path,
) -> None:
    metrics_df, failures_df = compute_metrics(
        target_id="test_id",
        ref_protein_pdb=protein_pdb_path,
        ref_ligand_sdf=ligand_sdf_path,
        pred_protein_pdb=predicted_protein_pdb_path,
        pred_ligand_sdf=predicted_ligand_sdf_path,
        output_dir=tmp_path,
    )

    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 1

    assert len(failures_df) == 0

    assert metrics_df["ligand_rmsd"][0] == pytest.approx(4.911, abs=1e-3)
    assert metrics_df["pocket_rmsd"][0] == pytest.approx(3.342, abs=1e-3)

    assert metrics_df["protein_tm_score"][0] == pytest.approx(0.758, abs=1e-3)
    assert metrics_df["protein_rmsd"][0] == pytest.approx(2.51, abs=1e-3)


def test_compute_metrics_with_input_parsing_errors(
    protein_pdb_path: Path,
    ligand_sdf_path: Path,
    predicted_protein_pdb_path: Path,
    predicted_ligand_sdf_path: Path,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "neuralplexer_benchmarks.metrics.all_metrics.rewrite_pdb_with_nplx_v3_input",
        side_effect=Exception("NPLXV3InputIOError"),
    )
    # Check that an exception is raised when return_exceptions is False
    with pytest.raises(ValueError, match="Exceptions occurred during PDB/SDF formatting"):
        compute_metrics(
            target_id="test_id",
            ref_protein_pdb=protein_pdb_path,
            ref_ligand_sdf=ligand_sdf_path,
            pred_protein_pdb=predicted_protein_pdb_path,
            pred_ligand_sdf=predicted_ligand_sdf_path,
            output_dir=tmp_path,
            return_exceptions=False,
        )

    metrics_df, failures_df = compute_metrics(
        target_id="test_id",
        ref_protein_pdb=protein_pdb_path,
        ref_ligand_sdf=ligand_sdf_path,
        pred_protein_pdb=predicted_protein_pdb_path,
        pred_ligand_sdf=predicted_ligand_sdf_path,
        output_dir=tmp_path,
        return_exceptions=True,
    )

    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 0

    assert len(failures_df) == 2

    assert failures_df["error_msg"][0].startswith(
        "NPLXV3InputIOError: Failed to rewrite reference PDB with NPLXV3Input"
    )
    assert failures_df["error_msg"][1].startswith(
        "NPLXV3InputIOError: Failed to rewrite predicted PDB/SDF with NPLXV3Input"
    )


def test_compute_metrics_with_errors(
    protein_pdb_path: Path,
    ligand_sdf_path: Path,
    predicted_protein_pdb_path: Path,
    predicted_ligand_sdf_path: Path,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "neuralplexer_benchmarks.metrics.all_metrics.compute_best_pocket_aligned_rmsd",
        side_effect=Exception("PocketAlignedRMSDError"),
    )
    mocker.patch(
        "neuralplexer_benchmarks.metrics.all_metrics.compute_protein_tm_score_and_rmsd",
        side_effect=Exception("TMScoreError"),
    )
    mocker.patch(
        "neuralplexer_benchmarks.metrics.all_metrics.run_posebusters_cmd",
        side_effect=Exception("PosebustersError"),
    )

    # Check that an exception is raised when return_exceptions is False
    with pytest.raises(ValueError, match="Exceptions occurred during metric computation"):
        compute_metrics(
            target_id="test_id",
            ref_protein_pdb=protein_pdb_path,
            ref_ligand_sdf=ligand_sdf_path,
            pred_protein_pdb=predicted_protein_pdb_path,
            pred_ligand_sdf=predicted_ligand_sdf_path,
            output_dir=tmp_path,
            return_exceptions=False,
        )

    # Check that exceptions are returned when return_exceptions is True
    metrics_df, failures_df = compute_metrics(
        target_id="test_id",
        ref_protein_pdb=protein_pdb_path,
        ref_ligand_sdf=ligand_sdf_path,
        pred_protein_pdb=predicted_protein_pdb_path,
        pred_ligand_sdf=predicted_ligand_sdf_path,
        output_dir=tmp_path,
        return_exceptions=True,
    )

    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 1

    assert len(failures_df) == 3
    assert failures_df["error_msg"][0] == "PocketAlignedRMSDError: PocketAlignedRMSDError"
    assert failures_df["error_msg"][1] == "TMScoreError: TMScoreError"
    assert failures_df["error_msg"][2] == "PosebustersError: PosebustersError"

from logging import Logger, LoggerAdapter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuralplexer_benchmarks.exceptions import (
    NPLXV3InputIOError,
    PocketAlignedRMSDError,
    PosebustersError,
    TMScoreError,
)
from neuralplexer_benchmarks.metrics.align_pocket import DEFAULT_POCKET_CUTOFF, compute_best_pocket_aligned_rmsd
from neuralplexer_benchmarks.metrics.posebusters_metrics import run_posebusters_cmd
from neuralplexer_benchmarks.metrics.tm_score import compute_protein_tm_score_and_rmsd
from neuralplexer_benchmarks.pdb_sdf_io import rewrite_ligand_sdf_with_reference_sdf, rewrite_pdb_with_nplx_v3_input

LOGGER = Logger(__name__)


def _compute_metrics(
    target_id: str,
    ref_protein_pdb: Path,
    ref_ligand_sdf: Path,
    pred_protein_pdb: Path,
    pred_ligand_sdf: Path,
    output_dir: Path,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    logger: Logger | LoggerAdapter = LOGGER,
) -> tuple[pd.DataFrame, list[Exception]]:
    metrics_dict: dict[str, Any] = {}
    exceptions: list[Exception] = []

    logger.info("Computing pocket-aligned RMSDs")

    try:
        aligned_ligand_sdf = output_dir / "pred_ligand_aligned.sdf"
        alignment_work_dir = output_dir / "alignment_work_dir"
        alignment_work_dir.mkdir(parents=True, exist_ok=True)

        ligand_rmsd, pocket_rmsd = compute_best_pocket_aligned_rmsd(
            ref_protein_pdb=ref_protein_pdb,
            ref_ligand_sdf=ref_ligand_sdf,
            pred_protein_pdb=pred_protein_pdb,
            pred_ligand_sdf=pred_ligand_sdf,
            aligned_ligand_sdf=aligned_ligand_sdf,
            work_dir=alignment_work_dir,
            pocket_cutoff=pocket_cutoff,
        )
        metrics_dict["ligand_rmsd"] = ligand_rmsd
        metrics_dict["pocket_rmsd"] = pocket_rmsd
        metrics_dict["rmsd_≤_2å"] = (ligand_rmsd < 2.0) if not pd.isna(ligand_rmsd) else np.nan
        metrics_dict["rmsd_≤_5å"] = (ligand_rmsd < 5.0) if not pd.isna(ligand_rmsd) else np.nan
    except Exception as e:
        logger.warning(f"PocketAlignedRMSDError: {e}")
        exceptions.append(PocketAlignedRMSDError(e))
        metrics_dict["ligand_rmsd"] = None
        metrics_dict["pocket_rmsd"] = None
        metrics_dict["rmsd_≤_2å"] = None
        metrics_dict["rmsd_≤_5å"] = None

    logger.info("Computing protein TM-score and RMSD")

    try:
        tm_score, rmsd = compute_protein_tm_score_and_rmsd(pdb_file=pred_protein_pdb, ref_pdb_file=ref_protein_pdb)
        metrics_dict["protein_tm_score"] = tm_score
        metrics_dict["protein_rmsd"] = rmsd
    except Exception as e:
        logger.warning(f"TMScoreError: {e}")
        exceptions.append(TMScoreError(e))
        metrics_dict["protein_tm_score"] = None
        metrics_dict["protein_rmsd"] = None

    logger.info("Computing posebusters metrics")

    try:
        posebusters_metrics_df = run_posebusters_cmd(
            predicted_ligand_sdf_path=pred_ligand_sdf,
            reference_ligand_sdf_path=ref_ligand_sdf,
            protein_pdb_path=pred_protein_pdb,
        )
        # This field is removed because it used the unaligned ligand
        # We use ligand rmsd computed via RDkit
        del posebusters_metrics_df["rmsd_≤_2å"]
        metrics_dict.update(posebusters_metrics_df.to_dict(orient="records")[0])
    except Exception as e:
        logger.warning(f"PosebustersError: {e}")
        exceptions.append(PosebustersError(e))

    metrics_df = pd.DataFrame(metrics_dict, index=[0]).fillna(np.nan)
    metrics_df.insert(0, "target_id", target_id)

    return metrics_df, exceptions


def compute_metrics(
    target_id: str,
    ref_protein_pdb: Path,
    ref_ligand_sdf: Path,
    pred_protein_pdb: Path,
    pred_ligand_sdf: Path,
    output_dir: Path,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    logger: Logger | LoggerAdapter = LOGGER,
    return_exceptions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all metrics for the given reference and predicted protein-ligand complexes.

    The reference and predicted protein PDB and ligand SDF files are rewritten with NPLXV3Input datamodel
    before computing the metrics.
    """
    exceptions: list[Exception] = []

    # Re-generate the reference pdb file with the NPLXV3Input format
    ref_protein_pdb_new = output_dir / "ref_protein_new.pdb"

    try:
        rewrite_pdb_with_nplx_v3_input(protein_pdb=ref_protein_pdb, output_protein_pdb=ref_protein_pdb_new)
    except Exception as e:
        message = f"Failed to rewrite reference PDB with NPLXV3Input: {e}"
        logger.warning(f"NPLXV3InputIOError: {message}")
        exceptions.append(NPLXV3InputIOError(message))

    # Re-generate the predicted pdb and sdf files with the NPLXV3Input format and reference ligand sdf
    pred_protein_pdb_new = output_dir / "pred_protein_new.pdb"
    pred_ligand_sdf_new = output_dir / "pred_ligand_new.sdf"

    try:
        rewrite_pdb_with_nplx_v3_input(protein_pdb=pred_protein_pdb, output_protein_pdb=pred_protein_pdb_new)
        rewrite_ligand_sdf_with_reference_sdf(
            pred_ligand_sdf=pred_ligand_sdf, ref_ligand_sdf=ref_ligand_sdf, output_ligand_sdf=pred_ligand_sdf_new
        )
    except Exception as e:
        message = f"Failed to rewrite predicted PDB/SDF with NPLXV3Input: {e}"
        logger.warning(f"NPLXV3InputIOError: {message}")
        exceptions.append(NPLXV3InputIOError(message))

    if len(exceptions) > 0:
        if return_exceptions:
            failures_df = pd.DataFrame(
                {"target_id": [target_id] * len(exceptions), "error_msg": [str(exc) for exc in exceptions]}
            )
            return pd.DataFrame(), failures_df
        else:
            raise ValueError(f"Exceptions occurred during PDB/SDF formatting: {exceptions}")

    metrics_df, additional_exceptions = _compute_metrics(
        target_id=target_id,
        ref_protein_pdb=ref_protein_pdb_new,
        ref_ligand_sdf=ref_ligand_sdf,
        pred_protein_pdb=pred_protein_pdb_new,
        pred_ligand_sdf=pred_ligand_sdf_new,
        output_dir=output_dir,
        pocket_cutoff=pocket_cutoff,
        logger=logger,
    )

    exceptions.extend(additional_exceptions)

    if not return_exceptions and len(exceptions) > 0:
        raise ValueError(f"Exceptions occurred during metric computation: {exceptions}")

    failures_df = pd.DataFrame(
        {"target_id": [target_id] * len(exceptions), "error_msg": [str(exc) for exc in exceptions]}
    )

    return metrics_df, failures_df

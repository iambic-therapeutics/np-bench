from collections import defaultdict
from logging import Logger, LoggerAdapter
from pathlib import Path

import numpy as np
import pandas as pd

from neuralplexer_benchmarks.exceptions import DockQError, GeneralizedRMSDError, TMScoreError
from neuralplexer_benchmarks.internal_evaluation.dockq_scores import compute_dockq_scores
from neuralplexer_benchmarks.internal_evaluation.generalized_rmsd import compute_generalized_ligand_rmsds
from neuralplexer_benchmarks.internal_evaluation.parse_interfaces import resolve_chain_and_interfaces
from neuralplexer_benchmarks.internal_evaluation.tm_scores import compute_tm_scores
from neuralplexer_benchmarks.logger import get_logger
from neuralplexer_benchmarks.metrics.align_pocket import DEFAULT_POCKET_CUTOFF
from neuralplexer_benchmarks.neuralplexer_data.datamodels import load_nplx_v3_input


def compute_internal_metrics(
    mmcif_id: str,
    chain_or_interface_ids: list[str],
    ref_npz_path: Path,
    pred_npz_path: Path,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    logger: Logger | LoggerAdapter | None = None,
    return_exceptions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute internal metrics for a single bioassembly based on local npz files.

    Args:
        mmcif_id (str): mmcif_id.
        chain_or_interface_ids (list[str]): List of chain or interface ids.
        ref_npz_local_path (Path): Path to the reference npz file.
        pred_npz_local_path (Path): Path to the predicted npz file.
        pocket_cutoff (float, optional): Pocket cutoff. Defaults to DEFAULT_POCKET_CUTOFF.
        logger ([type], optional): Logger. Defaults to None.
        return_exceptions (bool, optional): Whether to return exceptions. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of all_metrics_df and failures_df.
    """
    logger = get_logger(logger=logger)

    all_metrics_dict: dict[str, list] = defaultdict(list)
    exceptions: list[Exception] = []

    chain_and_interfaces = resolve_chain_and_interfaces(chain_or_interface_ids)

    ref_nplx_v3_input = load_nplx_v3_input(ref_npz_path)
    pred_nplx_v3_input = load_nplx_v3_input(pred_npz_path)

    logger.info("Computing DockQ scores")

    try:
        dockq_scores, best_chain_mapping = compute_dockq_scores(
            reference_nplx_v3_input=ref_nplx_v3_input,
            predicted_nplx_v3_input=pred_nplx_v3_input,
            polymer_polymer_interfaces=chain_and_interfaces.polymer_polymer_interfaces,
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"DockQError: {e}")
        exceptions.append(DockQError(e))
        dockq_scores = {}
        best_chain_mapping = None

    all_metrics_dict["chain_or_interface_id"] = list(dockq_scores.keys())
    all_metrics_dict["metrics"] = ["DockQ"] * len(dockq_scores)
    all_metrics_dict["value"] = list(dockq_scores.values())

    logger.info("Computing generalized pocket-aligned ligand RMSDs")

    try:
        ligand_rmsds = compute_generalized_ligand_rmsds(
            reference_nplx_v3_input=ref_nplx_v3_input,
            predicted_nplx_v3_input=pred_nplx_v3_input,
            ligand_polymer_interfaces=chain_and_interfaces.ligand_polymer_interfaces,
            chain_mapping=best_chain_mapping,
            pocket_cutoff=pocket_cutoff,
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"GeneralizedRMSDError: {e}")
        exceptions.append(GeneralizedRMSDError(e))
        ligand_rmsds = {}

    all_metrics_dict["chain_or_interface_id"].extend(list(ligand_rmsds.keys()))
    all_metrics_dict["metrics"].extend(["ligand_rmsd"] * len(ligand_rmsds))
    all_metrics_dict["value"].extend(list(ligand_rmsds.values()))

    logger.info("Computing TM scores")

    try:
        tm_scores = compute_tm_scores(
            reference_nplx_v3_input=ref_nplx_v3_input,
            predicted_nplx_v3_input=pred_nplx_v3_input,
            polymer_chain_ids=chain_and_interfaces.polymer_chain_ids,
            chain_mapping=best_chain_mapping,
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"TMScoreError: {e}")
        exceptions.append(TMScoreError(e))
        tm_scores = {}

    all_metrics_dict["chain_or_interface_id"].extend(list(tm_scores.keys()))
    all_metrics_dict["metrics"].extend(["tm_score"] * len(tm_scores))
    all_metrics_dict["value"].extend(list(tm_scores.values()))

    all_metrics_df = pd.DataFrame(all_metrics_dict).fillna(np.nan)
    all_metrics_df.insert(0, "mmcif_id", mmcif_id)

    if not return_exceptions and len(exceptions) > 0:
        raise ValueError(f"Exceptions occurred during metric computation: {exceptions}")

    failures_df = pd.DataFrame(
        {
            "mmcif_id": [mmcif_id] * len(exceptions),
            "chain_or_interface_id": [chain_or_interface_ids] * len(exceptions),
            "error_msg": [str(exc) for exc in exceptions],
        }
    )

    return all_metrics_df, failures_df

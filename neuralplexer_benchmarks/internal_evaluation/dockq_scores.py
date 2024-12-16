from logging import Logger, LoggerAdapter
from pathlib import Path
from tempfile import TemporaryDirectory

import more_itertools

from neuralplexer_benchmarks.metrics.dockq import ChainMapping, run_dockq_cli
from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input
from neuralplexer_benchmarks.neuralplexer_data.output import dump_nplx_v3_input_to_pdb_and_sdf

LOGGER = Logger(__name__)


def compute_dockq_scores(
    reference_nplx_v3_input: NPLXV3Input,
    predicted_nplx_v3_input: NPLXV3Input,
    polymer_polymer_interfaces: dict[tuple[str, str], str],
    logger: Logger | LoggerAdapter = LOGGER,
) -> tuple[dict[str, float], ChainMapping | None]:
    """
    Compute DockQ scores for the reference and predicted NPLXV3Input.

    Args:
        reference_nplx_v3_input (NPLXV3Input): Reference NPLXV3Input.
        predicted_nplx_v3_input (NPLXV3Input): Predicted NPLXV3Input.
        polymer_polymer_interfaces (dict[tuple[str, str], str]): Dictionary mapping polymer-polymer interfaces to interface ids.
        logger (Logger | LoggerAdapter, optional): Logger. Defaults to LOGGER.

    Returns:
        tuple[dict[str, float], ChainMapping | None]: Tuple containing the DockQ scores and the best chain mapping.
    """
    # Get the number of chains in the reference and predicted NPLXV3Input
    ref_chain_ids = list(reference_nplx_v3_input.chain_sequences.keys())
    pred_chain_ids = list(predicted_nplx_v3_input.chain_sequences.keys())

    ref_num_chains = len(ref_chain_ids)
    pred_num_chains = len(pred_chain_ids)

    if pred_num_chains != ref_num_chains:
        raise ValueError(f"Number of chains mismatch: reference {ref_num_chains}, predicted {pred_num_chains}")

    if set(ref_chain_ids) != set(pred_chain_ids):
        raise ValueError("The chain ids in the reference and predicted structures are not the same")

    if ref_num_chains == 1:
        logger.info("Single chain protein provided. Skipping the DockQ computation.")
        return {}, {more_itertools.one(ref_chain_ids): more_itertools.one(pred_chain_ids)}

    with TemporaryDirectory() as temp_dir:
        ref_pdb_path = Path(temp_dir) / "ref.pdb"
        pred_pdb_path = Path(temp_dir) / "pred.pdb"

        ref_pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(reference_nplx_v3_input, pdb_strategy="all")
        pred_pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(predicted_nplx_v3_input, pdb_strategy="all")

        ref_pdb_path.write_text(ref_pdb_str)
        pred_pdb_path.write_text(pred_pdb_str)

        # Run DockQ to compute DockQ scores
        dockq_result = run_dockq_cli(
            native_pdb_path=ref_pdb_path,
            model_pdb_path=pred_pdb_path,
            allowed_mismatches=0,
        )

    if dockq_result is None:
        return {}, None

    # check if the chain ids in the best mapping is a subset of the reference chain ids
    if not set(dockq_result.best_chain_mapping.keys()).issubset(set(ref_chain_ids)):
        raise ValueError("The chain ids in the DockQ best mapping are not a subset of the reference chain ids.")
    if not set(dockq_result.best_chain_mapping.values()).issubset(set(pred_chain_ids)):
        raise ValueError("The chain ids in the DockQ best mapping are not a subset of the predicted chain ids.")

    if not polymer_polymer_interfaces:
        logger.info("No polymer-polymer interfaces provided for DockQ score calculation.")
        return {}, dockq_result.best_chain_mapping

    dockq_scores: dict[str, float] = {}
    for native_pair, result in dockq_result.interface_results.items():
        if native_pair in polymer_polymer_interfaces:
            dockq_score = result["DockQ"]
            interface_id = polymer_polymer_interfaces[native_pair]
            dockq_scores[interface_id] = dockq_score

    return dockq_scores, dockq_result.best_chain_mapping

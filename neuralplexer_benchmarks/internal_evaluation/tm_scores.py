from logging import Logger, LoggerAdapter
from pathlib import Path
from tempfile import TemporaryDirectory

from neuralplexer_benchmarks.metrics.tm_score import compute_protein_tm_score_and_rmsd
from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input
from neuralplexer_benchmarks.neuralplexer_data.output import LigandStrategy, dump_nplx_v3_input_to_pdb_and_sdf

LOGGER = Logger(__name__)


def compute_tm_scores(
    reference_nplx_v3_input: NPLXV3Input,
    predicted_nplx_v3_input: NPLXV3Input,
    polymer_chain_ids: dict[str, str],
    chain_mapping: dict[str, str] | None,
    logger: Logger | LoggerAdapter = LOGGER,
) -> dict[str, float]:
    """
    Compute global and monomeric TM scores for the reference and predicted NPLXV3Input.

    Args:
        reference_nplx_v3_input (NPLXV3Input): Reference NPLXV3Input.
        predicted_nplx_v3_input (NPLXV3Input): Predicted NPLXV3Input.
        polymer_chain_ids (dict[str, str]): Dictionary mapping polymer chain ids to interface ids.
        chain_mapping (dict[str, str], optional): Dictionary mapping reference chain ids to predicted chain ids. Defaults to None.
        logger (Logger | LoggerAdapter, optional): Logger. Defaults to LOGGER.

    Returns:
        dict[str, float]: Dictionary containing the global and monomeric TM scores for each interface id.
    """
    tm_scores = {}
    with TemporaryDirectory() as temp_dir:
        ref_pdb_path = Path(temp_dir) / "ref.pdb"
        pred_pdb_path = Path(temp_dir) / "pred.pdb"

        ligand_strategy: LigandStrategy = LigandStrategy(residue_names=[])

        ref_chain_pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(
            reference_nplx_v3_input, ligand_strategy=ligand_strategy
        )
        pred_chain_pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(
            predicted_nplx_v3_input, ligand_strategy=ligand_strategy
        )

        ref_pdb_path.write_text(ref_chain_pdb_str)
        pred_pdb_path.write_text(pred_chain_pdb_str)

        logger.info("Computing global TM scores")

        global_tm_score, _ = compute_protein_tm_score_and_rmsd(pred_pdb_path, ref_pdb_path)

        tm_scores["global"] = global_tm_score

        logger.info("Computing monomeric TM scores")

        if not polymer_chain_ids or not chain_mapping:
            logger.info("No chain ids or best chain mapping provided. Skipping the monomeric TM score computation.")
            return tm_scores

        for ref_chain_id, pred_chain_id in chain_mapping.items():
            if ref_chain_id not in polymer_chain_ids:
                continue

            ref_chain_pdb_path = Path(temp_dir) / f"ref_{ref_chain_id}.pdb"
            pred_chain_pdb_path = Path(temp_dir) / f"pred_{pred_chain_id}.pdb"

            ref_chain_pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(
                reference_nplx_v3_input, pdb_strategy=[ref_chain_id]
            )
            pred_chain_pdb_str, _ = dump_nplx_v3_input_to_pdb_and_sdf(
                predicted_nplx_v3_input, pdb_strategy=[pred_chain_id]
            )

            ref_chain_pdb_path.write_text(ref_chain_pdb_str)
            pred_chain_pdb_path.write_text(pred_chain_pdb_str)

            monomer_tm_score, _ = compute_protein_tm_score_and_rmsd(pred_chain_pdb_path, ref_chain_pdb_path)

            polymer_chain_id = polymer_chain_ids[ref_chain_id]
            tm_scores[polymer_chain_id] = monomer_tm_score

    return tm_scores

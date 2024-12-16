import itertools
from collections import defaultdict
from logging import Logger, LoggerAdapter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from neuralplexer_benchmarks.exceptions import LigandUnboundError
from neuralplexer_benchmarks.metrics.align_pocket import (
    DEFAULT_POCKET_CUTOFF,
    compute_pocket_aligned_rmsds,
)
from neuralplexer_benchmarks.metrics.dockq import ChainMapping
from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input
from neuralplexer_benchmarks.neuralplexer_data.output import LigandStrategy, dump_nplx_v3_input_to_pdb_and_sdf

LOGGER = Logger(__name__)


def compute_generalized_ligand_rmsds(
    reference_nplx_v3_input: NPLXV3Input,
    predicted_nplx_v3_input: NPLXV3Input,
    ligand_polymer_interfaces: dict[tuple[str, str], str],
    chain_mapping: ChainMapping | None = None,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    logger: Logger | LoggerAdapter = LOGGER,
) -> dict[str, float | None]:
    """
    Compute the generalized pocket-aligned ligand RMSDs for the reference and predicted protein-ligand complexes.

    Args:
        reference_nplx_v3_input (NPLXV3Input): Reference NPLXV3Input.
        predicted_nplx_v3_input (NPLXV3Input): Predicted NPLXV3Input.
        ligand_polymer_interfaces (dict[tuple[str, str], str]): Dictionary mapping ligand-polymer interfaces to interface ids.
        pocket_cutoff (float, optional): Pocket cutoff. Defaults to DEFAULT_POCKET_CUTOFF.
        logger (Logger | LoggerAdapter, optional): Logger. Defaults to LOGGER.

    Returns:
        dict[str, float | None]: Dictionary containing the generalized pocket-aligned ligand RMSDs for each interface id.
    """
    if not ligand_polymer_interfaces:
        logger.info("No ligand-polymer interfaces provided. Skipping the computation.")
        return {}

    ligand_residue_names = _resolve_ligand_residue_names(ligand_polymer_interfaces)

    if len(ligand_residue_names) == 0:
        logger.info("No ligand residue names found in the ligand-polymer interfaces.")
        return {}

    logger.info(
        f"Found {len(ligand_residue_names)} ligand residue names in the ligand-polymer interfaces: "
        f"{ligand_residue_names}"
    )

    results = {}
    for ligand_residue_name in ligand_residue_names:
        interface_chain_ids = _get_interface_chain_ids(ligand_residue_name, ligand_polymer_interfaces)

        pred_chain_ids = [chain_mapping[chain_id] for chain_id in interface_chain_ids] if chain_mapping else None

        interface_chain_id_to_ligand_rmsds = _compute_ligand_rmsds_for_ligand_residue(
            reference_nplx_v3_input=reference_nplx_v3_input,
            predicted_nplx_v3_input=predicted_nplx_v3_input,
            ligand_residue_name=ligand_residue_name,
            interface_chain_ids=interface_chain_ids,
            pred_chain_ids=pred_chain_ids,
            pocket_cutoff=pocket_cutoff,
            logger=logger,
        )

        for interface_chain_id, ligand_rmsd in interface_chain_id_to_ligand_rmsds.items():
            interface_id = ligand_polymer_interfaces[(ligand_residue_name, interface_chain_id)]
            results[interface_id] = ligand_rmsd

    return results


def _compute_ligand_rmsds_for_ligand_residue(
    reference_nplx_v3_input: NPLXV3Input,
    predicted_nplx_v3_input: NPLXV3Input,
    ligand_residue_name: str,
    interface_chain_ids: list[str],
    pred_chain_ids: list[str] | None,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    logger: Logger | LoggerAdapter = LOGGER,
) -> dict[str, float | None]:
    logger.info(f"Computing pocket-aligned ligand RMSDs for ligand residue {ligand_residue_name}")

    if pred_chain_ids is not None and len(pred_chain_ids) != len(interface_chain_ids):
        raise ValueError(
            f"Number of predicted chain ids {len(pred_chain_ids)} does not match interface chain ids {len(interface_chain_ids)}"
        )

    ligand_strategy: LigandStrategy = LigandStrategy(residue_names=[ligand_residue_name])

    ref_pdb_str, ref_sdf_blocks = dump_nplx_v3_input_to_pdb_and_sdf(
        reference_nplx_v3_input, ligand_strategy=ligand_strategy
    )

    # Use all chain IDs in the reference and predicted structures to perform exhaustive pairwise alignments
    # and identify the minimum RMSD for each interface chain ID
    all_chain_ids = list(reference_nplx_v3_input.chain_sequences.keys())
    all_pred_chain_ids = list(predicted_nplx_v3_input.chain_sequences.keys())

    num_ref_ligands = len(ref_sdf_blocks)

    logger.info(f"Found {num_ref_ligands} ligands with residue name {ligand_residue_name} in the reference structure.")

    MAX_NUM_LIGANDS = 10

    # Limit the number of ligands to avoid excessive computation
    # TODO: Implement a more efficient way to handle a large number of ligands.
    # For example, we can use the distance of the ligand from the interface chain to filter out ligands that are too far away.
    if num_ref_ligands > MAX_NUM_LIGANDS:
        raise ValueError(
            f"Too many ligands ({num_ref_ligands}) with residue name {ligand_residue_name} in the reference structure."
        )

    pred_pdb_str, pred_sdf_blocks = dump_nplx_v3_input_to_pdb_and_sdf(
        predicted_nplx_v3_input, ligand_strategy=ligand_strategy
    )
    if len(pred_sdf_blocks) != num_ref_ligands:
        raise ValueError(
            f"Number of ligands mismatch: predicted {len(pred_sdf_blocks)} vs reference {num_ref_ligands}."
        )

    with TemporaryDirectory() as tmp_dir:
        this_work_dir = Path(tmp_dir)

        ref_pdb_path = this_work_dir / "ref_protein.pdb"
        ref_pdb_path.write_text(ref_pdb_str)

        pred_pdb_path = this_work_dir / "pred_protein.pdb"
        pred_pdb_path.write_text(pred_pdb_str)

        ref_ligand_sdf_paths = []
        for ligand_idx, ref_sdf_block in enumerate(ref_sdf_blocks):
            ref_ligand_sdf_path = this_work_dir / f"ref_ligand_{ligand_idx}.sdf"
            ref_ligand_sdf_path.write_text(ref_sdf_block)
            ref_ligand_sdf_paths.append(ref_ligand_sdf_path)

        pred_ligand_sdf_paths = []
        for ligand_idx, pred_sdf_block in enumerate(pred_sdf_blocks):
            pred_ligand_sdf_path = this_work_dir / f"pred_ligand_{ligand_idx}.sdf"
            pred_ligand_sdf_path.write_text(pred_sdf_block)
            pred_ligand_sdf_paths.append(pred_ligand_sdf_path)

        # Compute the pocket-aligned RMSDs for each pair permutation,
        # and store the minimum RMSD for each interface chain ID
        permutation_results: dict[str, list[float | None]] = defaultdict(list)

        ligand_is_bound = True

        # loop over pairs of reference and predicted ligands
        for ref_ligand_idx, pred_ligand_idx in itertools.product(range(num_ref_ligands), repeat=2):
            ref_ligand_sdf_path = ref_ligand_sdf_paths[ref_ligand_idx]
            pred_ligand_sdf_path = pred_ligand_sdf_paths[pred_ligand_idx]

            try:
                alignment_results = compute_pocket_aligned_rmsds(
                    ref_protein_pdb=ref_pdb_path,
                    ref_ligand_sdf=ref_ligand_sdf_path,
                    pred_protein_pdb=pred_pdb_path,
                    pred_ligand_sdf=pred_ligand_sdf_path,
                    work_dir=this_work_dir,
                    interface_chain_ids=all_chain_ids,
                    pred_chain_ids=all_pred_chain_ids,
                    pocket_cutoff=pocket_cutoff,
                )
            except LigandUnboundError as e:
                logger.warning(str(e))
                ligand_is_bound = False
                break

            ligand_rmsds = [res[0] if res is not None else None for res in alignment_results]

            for i, chain_id in enumerate(interface_chain_ids):
                permutation_results[chain_id].append(ligand_rmsds[i])

    if not ligand_is_bound:
        return {chain_id: np.inf for chain_id in interface_chain_ids}

    # Finding the minimum RMSD for each interface chain id across all permutations
    interface_chain_id_to_ligand_rmsds = {}
    for chain_id, ligand_rmsds in permutation_results.items():
        ligand_rmsds_without_none = [x for x in ligand_rmsds if x is not None]
        min_ligand_rmsd = min(ligand_rmsds_without_none) if len(ligand_rmsds_without_none) > 0 else None
        interface_chain_id_to_ligand_rmsds[chain_id] = min_ligand_rmsd

    return interface_chain_id_to_ligand_rmsds


def _resolve_ligand_residue_names(ligand_polymer_interfaces: dict[tuple[str, str], str]) -> list[str]:
    """
    Resolve the ligand residue names from the ligand-polymer interfaces.
    """
    if not ligand_polymer_interfaces:
        return []

    ligand_residue_names = set()
    for ligand_residue_name, _ in ligand_polymer_interfaces:
        ligand_residue_names.add(ligand_residue_name)
    return list(ligand_residue_names)


def _get_interface_chain_ids(
    ligand_residue_name: str, ligand_polymer_interfaces: dict[tuple[str, str], str]
) -> list[str]:
    """
    Get the interface chain IDs for the given ligand residue name.
    """
    interface_chain_ids = set()
    for ligand_residue_name_, interface_chain_id in ligand_polymer_interfaces:
        if ligand_residue_name_ == ligand_residue_name:
            interface_chain_ids.add(interface_chain_id)
    return sorted(interface_chain_ids)

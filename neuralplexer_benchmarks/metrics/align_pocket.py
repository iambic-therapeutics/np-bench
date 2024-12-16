import re
import shutil
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, check_output

import more_itertools
import numpy as np

from neuralplexer_benchmarks.exceptions import LigandUnboundError
from neuralplexer_benchmarks.metrics.ligand_rmsd import compute_ligand_symrmsd

# Default cutoff distance (in Angstroms) between any heavy atom of the ligand and the protein residues
DEFAULT_POCKET_CUTOFF = 10.0

LIGAND_CHAIN_ID = "x"
LIGAND_RESIDUE_NAME = "LG1"
UNKNOWN_CHAIN_ID = "U"

PYMOL_TIMEOUT_SECONDS = 300  # Timeout for the PyMOL CLI in seconds


def _run_pymol_command(pymol_command: str) -> str:
    """
    Run a PyMOL command using subprocess.

    Args:
        pymol_command (str): PyMOL command to run.

    Returns:
        str: Output of the PyMOL command.
    """
    cmd = ["pymol", "-c", "-q", "-d", pymol_command]
    try:
        result = check_output(cmd, shell=False, encoding="UTF-8", timeout=PYMOL_TIMEOUT_SECONDS)
    except TimeoutExpired:
        raise TimeoutError(f"PyMOL command timed out after {PYMOL_TIMEOUT_SECONDS} seconds.")
    except CalledProcessError as e:
        raise ValueError(f"PyMOL command failed with the following error: {e.stderr}")

    return result


def compute_best_pocket_aligned_rmsd(
    ref_protein_pdb: Path,
    ref_ligand_sdf: Path,
    pred_protein_pdb: Path,
    pred_ligand_sdf: Path,
    aligned_ligand_sdf: Path,
    work_dir: Path,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    ligand_residue_name: str = LIGAND_RESIDUE_NAME,
    ligand_chain_id: str = LIGAND_CHAIN_ID,
) -> tuple[float, float]:
    """
    Align two protein-ligand complexes by the pocket and compute the best RMSD of the pocket and ligand.

    The pocket is defined as the set of residues within `pocket_cutoff` of the ligand. For cases where
    there are multiple chains in the protein, the chain with the most residues in the pocket is used for alignment.
    The residues in this chain are then matched to the residues in the predicted complex for alignment.

    The RMSD of the pocket is computed using the heavy atoms of the residues in the pocket.
    The RMSD of the ligand is computed using the heavy atoms of the ligand.

    For cases where there are multiple chains in the protein, the alignment is performed for each chain separately.
    The chain with the lowest ligand RMSD is chosen as the optimal alignment.

    Args:
        ref_protein_pdb (Path): Path to the reference protein PDB file.
        ref_ligand_sdf (Path): Path to the reference ligand SDF file.
        pred_protein_pdb (Path): Path to the predicted protein PDB file.
        pred_ligand_sdf (Path): Path to the predicted ligand SDF file.
        aligned_ligand_sdf (Path): Path to save the aligned ligand SDF file.
        work_dir (Path): Path to the working directory.
        pocket_cutoff (float): Pocket cutoff distance.
        ligand_residue_name (str): Residue name of the ligand.
        ligand_chain_id (str): Chain ID of the ligand.

    Returns:
        tuple[float, float]: RMSD of the ligand and pocket.
    """

    result = more_itertools.one(
        compute_pocket_aligned_rmsds(
            ref_protein_pdb=ref_protein_pdb,
            ref_ligand_sdf=ref_ligand_sdf,
            pred_protein_pdb=pred_protein_pdb,
            pred_ligand_sdf=pred_ligand_sdf,
            work_dir=work_dir,
            pocket_cutoff=pocket_cutoff,
            ligand_residue_name=ligand_residue_name,
            ligand_chain_id=ligand_chain_id,
        )
    )

    if result is None:
        raise ValueError("No best pocket-aligned RMSD found between the reference and predicted complexes.")

    optimal_ligand_rmsd, optimal_pocket_rmsd, optimal_aligned_ligand_sdf = result

    shutil.copyfile(optimal_aligned_ligand_sdf, aligned_ligand_sdf)

    return optimal_ligand_rmsd, optimal_pocket_rmsd


def compute_pocket_aligned_rmsds(
    ref_protein_pdb: Path,
    ref_ligand_sdf: Path,
    pred_protein_pdb: Path,
    pred_ligand_sdf: Path,
    work_dir: Path,
    interface_chain_ids: list[str] | None = None,
    pred_chain_ids: list[str] | None = None,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    ligand_residue_name: str = LIGAND_RESIDUE_NAME,
    ligand_chain_id: str = LIGAND_CHAIN_ID,
) -> list[tuple[float, float, Path] | None]:
    """
    Compute the pocket-aligned RMSDs of the ligand for each chain in interface_chain_ids.

    if interface_chain_ids is None, the chain with the most residues in the pocket is used for alignment.
    otherwise, each chain in interface_chain_ids is used for alignment.

    if the chain in interface_chain_ids is not found in the pocket, the result is None.

    Args:
        ref_protein_pdb (Path): Path to the reference protein PDB file.
        ref_ligand_sdf (Path): Path to the reference ligand SDF file.
        pred_protein_pdb (Path): Path to the predicted protein PDB file.
        pred_ligand_sdf (Path): Path to the predicted ligand SDF file.
        work_dir (Path): Path to the working directory.
        interface_chain_ids (list[str], optional): List of chain IDs to use for alignment. Defaults to None.
        pocket_cutoff (float): Pocket cutoff distance.
        ligand_residue_name (str): Residue name of the ligand.
        ligand_chain_id (str): Chain ID of the ligand.

    Returns:
        list[tuple[float, float, Path] | None]: List of tuples of (ligand RMSD, pocket RMSD, aligned ligand SDF) or None.
    """
    # Get the residue IDs within the pocket around the ligand in the reference complex
    # These residues will be matched in the predicted complex and used for alignment
    ref_pocket_pdb = work_dir / "ref_pocket.pdb"
    ref_pocket_chain_and_residues = _get_pocket_chain_and_residue_ids(
        protein_pdb=ref_protein_pdb,
        ligand_sdf=ref_ligand_sdf,
        pocket_pdb=ref_pocket_pdb,
        pocket_cutoff=pocket_cutoff,
        ligand_residue_name=ligand_residue_name,
        ligand_chain_id=ligand_chain_id,
    )

    if interface_chain_ids is None:
        interface_chain_ids = [_get_chain_id_with_most_residues(ref_pocket_chain_and_residues)]

    if pred_chain_ids is None:
        # If a particular permutation of chains is not provided, we need to iterate over all pocket chains
        # in the predicted complex to find the best alignment.
        # This is because the chain ordering in the predicted complex may not be the same as
        # the chain ordering in the reference complex.
        pred_pocket_pdb = work_dir / "pred_pocket.pdb"
        pred_pocket_chain_and_residues = _get_pocket_chain_and_residue_ids(
            protein_pdb=pred_protein_pdb,
            ligand_sdf=pred_ligand_sdf,
            pocket_pdb=pred_pocket_pdb,
            pocket_cutoff=pocket_cutoff,
            ligand_residue_name=ligand_residue_name,
            ligand_chain_id=ligand_chain_id,
        )
        pred_chain_ids = list(pred_pocket_chain_and_residues.keys())

    if len(pred_chain_ids) == 0:
        raise LigandUnboundError(ligand_residue_name=ligand_residue_name, pocket_cutoff=pocket_cutoff)

    results: list[tuple[float, float, Path] | None] = []
    for interface_chain_id in interface_chain_ids:
        if interface_chain_id in ref_pocket_chain_and_residues:
            interface_chain_residues = list(ref_pocket_chain_and_residues[interface_chain_id])
            result = _compute_best_pocket_aligned_rmsd_for_chain_id(
                ref_protein_pdb=ref_protein_pdb,
                ref_ligand_sdf=ref_ligand_sdf,
                pred_protein_pdb=pred_protein_pdb,
                pred_ligand_sdf=pred_ligand_sdf,
                ref_chain_id=interface_chain_id,
                ref_chain_residues=interface_chain_residues,
                pred_chain_ids=pred_chain_ids,
                work_dir=work_dir,
                pocket_cutoff=pocket_cutoff,
                ligand_residue_name=ligand_residue_name,
            )
            results.append(result)
        else:
            results.append(None)

    return results


def _compute_best_pocket_aligned_rmsd_for_chain_id(
    ref_protein_pdb: Path,
    ref_ligand_sdf: Path,
    pred_protein_pdb: Path,
    pred_ligand_sdf: Path,
    ref_chain_id: str,
    ref_chain_residues: list[str],
    pred_chain_ids: list[str],
    work_dir: Path,
    pocket_cutoff: float,
    ligand_residue_name: str,
) -> tuple[float, float, Path]:
    """
    Compute the optimal pocket-aligned RMSD for a given reference chain ID.

    The optimal alignment is chosen based on the minimum ligand RMSD among the predicted chains.
    """
    ref_chain_residue_selection = _generate_pymol_residue_selection_string(ref_chain_residues)

    # Align the predicted complex to the reference comples by the pocket for each chain
    pocket_rmsd_by_chain = []
    ligand_rmsd_by_chain = []
    aligned_complex_pdb_by_chain = []
    aligned_ligand_sdf_by_chain = []
    for chain_id in pred_chain_ids:
        aligned_complex_pdb_this_chain = work_dir / f"complex_aligned_{chain_id}.pdb"
        aligned_ligand_sdf_this_chain = work_dir / f"ligand_aligned_{chain_id}.sdf"

        selected_residues = f"chain {chain_id} and resi {ref_chain_residue_selection}"

        pocket_rmsd = _align_pocket_by_chain(
            ref_protein_pdb=ref_protein_pdb,
            ref_ligand_sdf=ref_ligand_sdf,
            pred_protein_pdb=pred_protein_pdb,
            pred_ligand_sdf=pred_ligand_sdf,
            ref_chain_id=ref_chain_id,
            selected_residues=selected_residues,
            aligned_complex_pdb=aligned_complex_pdb_this_chain,
            aligned_ligand_sdf=aligned_ligand_sdf_this_chain,
            pocket_cutoff=pocket_cutoff,
            ligand_residue_name=ligand_residue_name,
        )

        ligand_rmsd = compute_ligand_symrmsd(ref_ligand_sdf, aligned_ligand_sdf_this_chain)

        pocket_rmsd_by_chain.append(pocket_rmsd)
        ligand_rmsd_by_chain.append(ligand_rmsd)
        aligned_complex_pdb_by_chain.append(aligned_complex_pdb_this_chain)
        aligned_ligand_sdf_by_chain.append(aligned_ligand_sdf_this_chain)

    min_rmsd_idx = np.argmin(ligand_rmsd_by_chain)

    optimal_pocket_rmsd = pocket_rmsd_by_chain[min_rmsd_idx]
    optimal_ligand_rmsd = ligand_rmsd_by_chain[min_rmsd_idx]
    optimal_aligned_ligand_sdf = aligned_ligand_sdf_by_chain[min_rmsd_idx]

    return optimal_ligand_rmsd, optimal_pocket_rmsd, optimal_aligned_ligand_sdf


def _get_pocket_chain_and_residue_ids(
    protein_pdb: Path,
    ligand_sdf: Path,
    pocket_pdb: Path,
    pocket_cutoff: float,
    ligand_residue_name: str,
    ligand_chain_id: str,
) -> dict[str, set[str]]:
    """
    Get the chain and residue IDs within a pocket around the ligand.

    Args:
        protein_pdb (Path): Path to the protein PDB file.
        ligand_sdf (Path): Path to the ligand SDF file.
        pocket_pdb (Path): Path to save the pocket residues PDB file.
        pocket_cutoff (float): Pocket cutoff distance.
        ligand_residue_name (str): Residue name of the ligand.
        ligand_chain_id (str): Chain ID of the ligand.

    Returns:
        dict[str, set[str]]: Dictionary of {chain ID: residue IDs} within the pocket.
    """
    # Construct the PyMOL command
    pymol_command = f"""
load {str(protein_pdb)}, protein;
load {str(ligand_sdf)}, ligand;
alter ligand, resn='{ligand_residue_name}';
create complex, protein or ligand;
alter resn {ligand_residue_name}, chain='{ligand_chain_id}';

select pocket_residues, byres (complex within {pocket_cutoff} of (resname {ligand_residue_name} and complex));
save {str(pocket_pdb)}, pocket_residues;
quit;
"""

    # Execute the PyMOL command and capture output and error
    _run_pymol_command(pymol_command)

    # Read the pocket residues from the PDB file
    chain_id_to_residues: dict[str, set[str]] = {}
    with open(pocket_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]

                # exclude ligand chain from the pocket residues
                if chain_id == ligand_chain_id:
                    continue

                # if chain ID is empty, set it to "U" (unknown)
                if chain_id == " ":
                    chain_id = UNKNOWN_CHAIN_ID

                if chain_id not in chain_id_to_residues:
                    chain_id_to_residues[chain_id] = set()

                residue_number = line[22:27].strip()
                chain_id_to_residues[chain_id].add(residue_number)

    return chain_id_to_residues


def _get_chain_id_with_most_residues(chain_id_to_residues: dict[str, set[str]]) -> str:
    """
    Get the chain ID with the most residues in the pocket.
    """
    # find the chain id with the most residues
    max_residues = 0
    max_chain_id = ""
    for chain_id, residues in chain_id_to_residues.items():
        if len(residues) > max_residues:
            max_residues = len(residues)
            max_chain_id = chain_id

    return max_chain_id


def _generate_pymol_residue_selection_string(residues: list[str]) -> str:
    """
    Generate a PyMOL residue selection string based on the residues in the pocket and the ligand.

    Args:
        chain_id_to_residues (dict[str, set[str]]): Dictionary of {chain ID: residue IDs} within the pocket.

    Returns:
        tuple[str, str]: Chain ID with the most residues, and the PyMOL residue selection string.
    """
    residue_selection_string = "+".join(sorted(residues))
    return residue_selection_string


def _align_pocket_by_chain(
    ref_protein_pdb: Path,
    ref_ligand_sdf: Path,
    pred_protein_pdb: Path,
    pred_ligand_sdf: Path,
    ref_chain_id: str,
    selected_residues: str,
    aligned_complex_pdb: Path,
    aligned_ligand_sdf: Path,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    ligand_residue_name: str = LIGAND_RESIDUE_NAME,
    ligand_chain_id: str = LIGAND_CHAIN_ID,
) -> float:
    """
    Align the predicted complex to the reference complex by the pocket residues.

    Args:
        ref_protein_pdb (Path): Path to the reference protein PDB file.
        ref_ligand_sdf (Path): Path to the reference ligand SDF file.
        pred_protein_pdb (Path): Path to the predicted protein PDB file.
        pred_ligand_sdf (Path): Path to the predicted ligand SDF file.
        ref_chain_id (str): Chain ID of the reference protein.
        selected_residues (str): PyMOL residue selection string for the pocket residues.
        aligned_complex_pdb (Path): Path to save the aligned complex PDB file.
        aligned_ligand_sdf (Path): Path to save the aligned ligand SDF file.
        pocket_cutoff (float): Pocket cutoff distance.
        ligand_residue_name (str): Residue name of the ligand.
        ligand_chain_id (str): Chain ID of the ligand.

    Returns:
        float: RMSD of the pocket residues.
    """
    # Construct the PyMOL command for alignment
    pymol_command = f"""
load {str(ref_protein_pdb)}, ref_protein;
load {str(ref_ligand_sdf)}, ref_ligand;
alter ref_ligand, resn='{ligand_residue_name}';
create ref_complex, ref_protein or ref_ligand;
alter resn {ligand_residue_name}, chain='{ligand_chain_id}';

load {str(pred_protein_pdb)}, pred_complex;
load {str(pred_ligand_sdf)}, pred_ligand;
alter pred_ligand, resn='{ligand_residue_name}';
create pred_complex, pred_complex or pred_ligand;
alter resn {ligand_residue_name}, chain='{ligand_chain_id}';

select ref_pocket, byres (ref_complex within {pocket_cutoff} of (resname {ligand_residue_name} and ref_complex));

select ref_single_chain_residues, (chain {ref_chain_id} and ref_pocket and name CA) or (chain {ligand_chain_id} and resn {ligand_residue_name} and ref_pocket);

select matching_residues, ({selected_residues} and pred_complex and name CA) or (chain {ligand_chain_id} and resn {ligand_residue_name} and pred_complex);

align matching_residues, ref_single_chain_residues, cycles=0;

save {str(aligned_complex_pdb)}, pred_complex;
create aligned_ligand, pred_complex and resname {ligand_residue_name};
save {str(aligned_ligand_sdf)}, aligned_ligand;
quit;
"""

    # Execute the PyMOL command and capture output and error
    output = _run_pymol_command(pymol_command)

    # extract the RMSD from the PyMOL output
    rmsd = _extract_pocket_rmsd(output)

    return rmsd


def _extract_pocket_rmsd(pymol_output: str) -> float:
    """
    Extract the RMSD of the pocket residues from the PyMOL output.
    """
    rmsd_pattern = re.compile(r"Executive: RMSD =\s+([0-9.]+)")

    rmsd_pattern_match = rmsd_pattern.search(pymol_output)
    if rmsd_pattern_match:
        rmsd = float(rmsd_pattern_match.group(1))
        return rmsd
    else:
        raise ValueError("RMSD not found in PyMOL output, likely due to an error in the PyMOL command.")

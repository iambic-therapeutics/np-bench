import itertools
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from subprocess import TimeoutExpired, run
from typing import Any, no_type_check

from DockQ.DockQ import (
    count_chain_combinations,
    format_mapping,
    format_mapping_string,
    get_all_chain_maps,
    group_chains,
    load_PDB,
    run_on_all_native_interfaces,
)
from parallelbar import progress_map

from neuralplexer_benchmarks.logger import get_logger
from neuralplexer_benchmarks.utils import get_num_procs

DOCKQ_TIMEOUT_SECONDS = 300  # Timeout for the DockQ CLI in seconds


InterfaceResults = dict[tuple[str, str], dict]
ChainMapping = dict[str, str]


@dataclass
class DockQResult:
    total_dockq: float
    best_chain_mapping: ChainMapping
    interface_results: InterfaceResults


def run_dockq_cli(
    native_pdb_path: Path,
    model_pdb_path: Path,
    mapping: str | None = None,
    capri_peptide: bool = False,
    small_molecule: bool = False,
    no_align: bool = False,
    allowed_mismatches: int = 0,
    n_cpu: int | None = None,
    max_chunk: int = 512,
) -> DockQResult | None:
    """
    Run DockQ CLI to compute DockQ scores.

    The DockQ CLI only outputs the best DockQ score and the corresponding chain mapping.
    """
    if not native_pdb_path.exists() or not model_pdb_path.exists():
        raise FileNotFoundError("Native or model PDB file does not exist.")

    n_cpu = n_cpu or get_num_procs()

    cmd = [
        "DockQ",
        str(model_pdb_path.absolute()),  # model
        str(native_pdb_path.absolute()),  # native
        "--n_cpu",
        str(n_cpu),
        "--max_chunk",
        str(max_chunk),
        "--allowed_mismatches",
        str(allowed_mismatches),
    ]
    if mapping:
        cmd += ["--mapping", mapping]
    if capri_peptide:
        cmd += ["--capri_peptide"]
    if small_molecule:
        cmd += ["--small_molecule"]
    if no_align:
        cmd += ["--no_align"]

    try:
        result = run(cmd, shell=False, check=True, text=True, capture_output=True, timeout=DOCKQ_TIMEOUT_SECONDS)  # nosec B603
    except TimeoutExpired:
        raise TimeoutError(f"DockQ command timed out after {DOCKQ_TIMEOUT_SECONDS} seconds.")

    if result.returncode != 0:
        raise ValueError(f"DockQ command failed with the following error: {result.stderr}")

    return _parse_dockq_cli_output(result.stdout)


def _parse_dockq_cli_output(output: str) -> DockQResult | None:
    """
    Parse the output of the DockQ CLI.

    The output is expected to be in the following format:
    ```
    Total DockQ over 3 native interfaces: 0.653 with BAC:ABC model:native mapping
    Native chains: A, B
        Model chains: B, A
        DockQ: 0.994
        irms: 0.000
    ```

    Args:
        output (str): Output of the DockQ CLI.

    Returns:
        tuple[dict[tuple[str, str], dict], dict[str, str] | None]: Dictionary containing the DockQ scores for each
        native chain pair and the best chain mapping.
    """
    if "Need at least two chains in the two inputs" in output:
        return None

    # Splitting the output into lines
    lines = output.strip().split("\n")

    # Initializing variables
    interface_results: dict[tuple[str, str], dict] = {}
    total_dockq = None
    best_mapping = None

    current_native_chains = None
    parsed_values: dict[str, Any] = {}
    in_native_chains_block = False

    for line in lines:
        line = line.strip()

        # find the best mapping using regex
        if line.startswith("Total DockQ over"):
            total_dockq = float(line.split("native interfaces:")[1].split(" with ")[0])
            best_mapping_str = line.split(" with ")[1].split(" model:native mapping")[0]
            best_mapping = _extract_chain_mapping_from_str(best_mapping_str)

        # Checking if the line indicates native chains
        if line.startswith("Native chains"):
            in_native_chains_block = True

            if current_native_chains:
                # Save the previous parsed values under the previous native chains
                interface_results[current_native_chains] = parsed_values

            # Parse the new native chains
            chain_a, chain_b = line.split(": ")[1].strip().split(", ")
            current_native_chains = (chain_a, chain_b)
            parsed_values = {}
        elif in_native_chains_block and line.strip() and ":" in line:
            # Parse key-value pairs
            key, value = line.split(": ")
            if key.strip() == "Model chains":
                parsed_values[key.strip()] = tuple(value.strip().split(", "))
            else:
                parsed_values[key.strip()] = float(value.strip())

    # Adding the last parsed values
    if current_native_chains:
        interface_results[current_native_chains] = parsed_values

    if total_dockq is None or best_mapping is None:
        return None
    else:
        return DockQResult(
            total_dockq=total_dockq,
            best_chain_mapping=best_mapping,
            interface_results=interface_results,
        )


def _extract_chain_mapping_from_str(mapping_str: str) -> ChainMapping:
    """
    Extract the chain mapping from the DockQ mapping string.

    The mapping string is expected to be in the following format (model:native mapping):
    ```
    BAC:ABC
    ```
    Converts the mapping string into a dictionary where the keys are the native chains and the values are the model chains.

    Args:
        mapping_str (str): DockQ mapping string.

    Returns:
        dict[str, str]: Dictionary containing the chain mapping.
    """
    model_chains, native_chains = mapping_str.split(":")
    return dict(zip(native_chains, model_chains))


####################################################################################################
### The following code is adapted from the DockQ.DockQ.main() function from v2.1.1 release.
### https://github.com/bjornwallner/DockQ/blob/3b03ce135975012fd87e99d655aeeceacbc46339/src/DockQ/DockQ.py#L864
###
### The code may not be synced the current version of DockQ. Use with caution.
####################################################################################################
@no_type_check
def run_dockq_python_api(
    native_pdb_path: Path,
    model_pdb_path: Path,
    mapping: str | None = None,
    capri_peptide: bool = False,
    small_molecule: bool = False,
    no_align: bool = False,
    allowed_mismatches: int = 0,
    n_cpu: int | None = None,
    max_chunk: int = 512,
) -> dict[str, Any]:
    """
    Python API to run DockQ.

    This is adapted from the DockQ.DockQ.main() function:
    https://github.com/bjornwallner/DockQ/blob/3b03ce135975012fd87e99d655aeeceacbc46339/src/DockQ/DockQ.py#L864

    With the following changes:
    - Replaced the command-line argument parsing with function arguments.
    - Removed the printing of the results.
    - Added the return of the results as a dictionary.
    - Added logging of the DockQ run.

    Other than that, the code should be functionally equivalent to the original DockQ main() function.

    Args:
        model (str): Path to the model PDB file.
        native (str): Path to the native PDB file.
        mapping (str, optional): Path to the mapping file. Defaults to None.
        capri_peptide (bool, optional): Use CAPRI peptide mode. Defaults to False.
        small_molecule (bool, optional): Use small molecule mode. Defaults to False.
        no_align (bool, optional): Skip alignment of the interfaces. Defaults to False.
        allowed_mismatches (int, optional): Number of allowed mismatches. Defaults to 0.
        max_chunk (int, optional): Maximum chunk size for parallel processing. Defaults to 512.

    Returns:
        dict[str, Any]: Dictionary containing the DockQ results.
    """
    logger = get_logger()

    if not native_pdb_path.exists() or not model_pdb_path.exists():
        raise FileNotFoundError("Native or model PDB file does not exist.")

    model = str(model_pdb_path.absolute())
    native = str(native_pdb_path.absolute())

    logger.info("Running DockQ python interface")

    initial_mapping, model_chains, native_chains = format_mapping(mapping, small_molecule)

    model_structure = load_PDB(model, chains=model_chains, small_molecule=small_molecule)
    native_structure = load_PDB(native, chains=native_chains, small_molecule=small_molecule)

    # check user-given chains are in the structures
    model_chains = [c.id for c in model_structure] if not model_chains else model_chains
    native_chains = [c.id for c in native_structure] if not native_chains else native_chains

    if len(model_chains) < 2 or len(native_chains) < 2:
        logger.info("DockQ needs at least two chains in the two inputs\n")
        return {}

    # permute chains and run on a for loop
    best_dockq = -1
    best_result = None
    best_mapping = None

    model_chains_to_combo = [mc for mc in model_chains if mc not in initial_mapping.values()]
    native_chains_to_combo = [nc for nc in native_chains if nc not in initial_mapping.keys()]

    chain_clusters, reverse_map = group_chains(
        model_structure,
        native_structure,
        model_chains_to_combo,
        native_chains_to_combo,
        allowed_mismatches,
    )

    chain_maps = get_all_chain_maps(
        chain_clusters,
        initial_mapping,
        reverse_map,
        model_chains_to_combo,
        native_chains_to_combo,
    )

    num_chain_combinations = count_chain_combinations(chain_clusters)
    # copy iterator to use later
    chain_maps, chain_maps_ = itertools.tee(chain_maps)

    low_memory = num_chain_combinations > 100

    run_chain_map = partial(
        run_on_all_native_interfaces,
        model_structure,
        native_structure,
        no_align=no_align,
        capri_peptide=capri_peptide,
        low_memory=low_memory,
    )

    if num_chain_combinations > 1:
        n_cpu = n_cpu or get_num_procs()
        cpus = min(num_chain_combinations, n_cpu)
        chunk_size = min(max_chunk, max(1, num_chain_combinations // cpus))

        # for large num_chain_combinations it should be possible to divide the chain_maps in chunks
        result_this_mappings = progress_map(
            run_chain_map,
            chain_maps,
            total=num_chain_combinations,
            n_cpu=cpus,
            chunk_size=chunk_size,
        )

        for chain_map, (result_this_mapping, total_dockq) in zip(chain_maps_, result_this_mappings):
            if total_dockq > best_dockq:
                best_dockq = total_dockq
                best_result = result_this_mapping
                best_mapping = chain_map

        if low_memory:
            # retrieve the full output by rerunning the best chain mapping
            best_result, total_dockq = run_on_all_native_interfaces(
                model_structure,
                native_structure,
                chain_map=best_mapping,
                no_align=no_align,
                capri_peptide=capri_peptide,
                low_memory=False,
            )

    else:  # skip multi-threading for single jobs (skip the bar basically)
        best_mapping = next(chain_maps)
        best_result, best_dockq = run_chain_map(best_mapping)

    info = {}
    info["model"] = model
    info["native"] = native
    info["best_dockq"] = best_dockq
    info["best_result"] = best_result
    info["GlobalDockQ"] = best_dockq / len(best_result)
    info["best_mapping"] = best_mapping
    info["best_mapping_str"] = f"{format_mapping_string(best_mapping)}"

    return info

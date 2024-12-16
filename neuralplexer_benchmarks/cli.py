import random
import sys
from collections import defaultdict
from logging import Logger, LoggerAdapter, StreamHandler
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from more_itertools import one

from neuralplexer_benchmarks.internal_evaluation.internal_metrics import compute_internal_metrics
from neuralplexer_benchmarks.metrics.align_pocket import DEFAULT_POCKET_CUTOFF
from neuralplexer_benchmarks.metrics.all_metrics import compute_metrics
from neuralplexer_benchmarks.neuralplexer_data.cif_utils import load_mm_cif
from neuralplexer_benchmarks.neuralplexer_data.datamodels import dump_nplx_v3_input
from neuralplexer_benchmarks.neuralplexer_data.pdb_sdf_io import load_pdb_and_sdfs
from neuralplexer_benchmarks.plotting import plot_combined_stats

LOGGER = Logger(__name__)
LOGGER.addHandler(StreamHandler(stream=sys.stdout))

app = typer.Typer(rich_markup_mode="rich")


def get_ref_protein_pdb_path(dataset_dir: str, target_id: str) -> str:
    return f"{dataset_dir}/{target_id}/{target_id}_protein.pdb"


def get_ref_ligand_sdf_path(dataset_dir: str, target_id: str) -> str:
    return f"{dataset_dir}/{target_id}/{target_id}_ligand.sdf"


def get_pred_protein_pdb_path(predictions_folder: Path | str, target_id: str, conf_idx: int | str) -> str:
    return f"{predictions_folder}/{target_id}/{conf_idx}/prot.pdb"


def get_pred_ligand_sdf_path(predictions_folder: Path | str, target_id: str, conf_idx: int | str) -> str:
    return f"{predictions_folder}/{target_id}/{conf_idx}/lig_0.sdf"


def get_all_mmcif_ids_from_dataset_index_df(dataset_index_df: pd.DataFrame) -> list[str]:
    return dataset_index_df["mmcif_id"].unique().tolist()


def get_ref_cif_path(dataset_folder: Path | str, mmcif_id: str) -> Path:
    return Path(f"{dataset_folder}/{mmcif_id}_eval.cif")


def get_pred_sample_folder(predictions_folder: Path | str, mmcif_id: str) -> str:
    return f"{predictions_folder}/{mmcif_id}"


def get_pred_conf_folder(predictions_folder: Path | str, mmcif_id: str, conf_idx: int | str) -> str:
    sample_folder = get_pred_sample_folder(predictions_folder, mmcif_id)
    return f"{sample_folder}/{conf_idx}"


def get_pred_cif_or_pdb_path(predictions_folder: Path | str, mmcif_id: str, conf_idx: int | str) -> Path:
    """
    Get the path to the predicted cif file for a given mmcif_id and conformation index.
    """
    conf_folder = get_pred_conf_folder(predictions_folder, mmcif_id, conf_idx)
    # Try to find the output.cif file, otherwise return the prot.pdb file
    conf_cif_path = Path(f"{conf_folder}/output.cif")
    if conf_cif_path.is_file():
        return conf_cif_path
    else:
        conf_pdb_path = Path(f"{conf_folder}/prot.pdb")
        if conf_pdb_path.is_file():
            return conf_pdb_path
    return conf_cif_path


def get_chain_or_interface_ids(mmcif_id: str, dataset_index_df: pd.DataFrame) -> list[str]:
    this_mmcif_id_df = dataset_index_df[dataset_index_df["mmcif_id"] == mmcif_id]
    interface_ids = this_mmcif_id_df["this_interface_id"].dropna().tolist()
    chain_ids = this_mmcif_id_df["this_chain_or_ccd_id"].dropna().tolist()
    return interface_ids + chain_ids


def get_conformer_metrics_path(output_folder: str) -> str:
    return f"{output_folder}/metrics.csv"


def get_conformer_failure_path(output_folder: str) -> str:
    return f"{output_folder}/failure.csv"


@app.command(name="posebusters")
def run_posebusters_benchmark(
    dataset_folder: str = typer.Option(..., "-d", "--dataset", help="Path to the Posebusters dataset folder."),
    predictions_folder: str = typer.Option(..., "-p", "--predictions", help="Path to the predictions folder."),
    num_conf: Optional[int] = typer.Option(None, "-n", "--num-conf", help="Number of conformations to evaluate."),
    conf_idx: Optional[int] = typer.Option(None, "-c", "--conf-idx", help="Conformer index to evaluate."),
    score_top_ranked: bool = typer.Option(False, "--score-top-ranked", help="Score the top ranked conformations."),
    use_cache: bool = typer.Option(False, "--use-cache", help="Use cached results if available."),
) -> None:
    """
    Run local benchmarking on Posebusters-like dataset.

    The dataset folder must have the following structure:

    dataset_folder/
    ├── target_1/
    │   ├── target_1_protein.pdb
    │   └── target_1_ligand.sdf
    ├── target_2/
    │   ├── target_2_protein.pdb
    │   └── target_2_ligand.sdf
    ├── ...

    The predictions folder must have the following structure:

    predictions_folder/
    ├── target_1/
    │   ├── conf_0/
    │   │   ├── prot.pdb
    │   │   └── lig_0.sdf
    │   ├── conf_1/
    │   │   ├── prot.pdb
    │   │   └── lig_0.sdf
    |   ├── ...
    ├── target_2/
    │   ├── conf_0/
    │   │   ├── prot.pdb
    │   │   └── lig_0.sdf
    │   ├── conf_1/
    │   │   ├── prot.pdb
    │   │   └── lig_0.sdf
    |   |-- ...
    ├── ...
    """
    metrics_path = _run_posebusters_benchmark(
        dataset_folder=dataset_folder,
        predictions_folder=predictions_folder,
        num_conf=num_conf,
        conf_idx=conf_idx,
        score_top_ranked=score_top_ranked,
        use_cache=use_cache,
    )
    typer.echo(f"Metrics saved to {metrics_path}")


@app.command(name="recent-pdb-eval", help="Run local benchmarking on recent PDB evaluation targets.")
def run_recent_pdb_eval_benchmark(
    dataset_folder: str = typer.Option(..., "-d", "--dataset", help="Path to the Recent PDB Evaluation Set."),
    dataset_index_path: str = typer.Option(
        "results/recent_pdb_eval_set_v2_w_CASP15RNA_reduced.csv",
        "-i",
        "--index",
        help="Path to the dataset index csv file.",
    ),
    predictions_folder: str = typer.Option(..., "-p", "--predictions", help="Path to the prediction cif or pdb files."),
    num_conf: Optional[int] = typer.Option(None, "-n", "--num-conf", help="Number of conformations to evaluate."),
    conf_idx: Optional[int] = typer.Option(None, "-c", "--conf-idx", help="Conformer index to evaluate."),
    score_top_ranked: bool = typer.Option(False, "--score-top-ranked", help="Score the top ranked conformations."),
    use_cache: bool = typer.Option(False, "--use-cache", help="Use cached results if available."),
) -> None:
    """
    Run local benchmarking on Recent PDB Evaluation Set.

    The dataset folder must have the following structure:

    dataset_folder/
    ├── mmcif_id_1_eval.cif
    ├── mmcif_id_2_eval.cif
    ├── ...

    The predictions folder must have the following structure:

    predictions_folder/
    |── mmcif_id_1/
    |   ├── conf_0/
    |   │   ├── output.cif # or output.pdb
    |   ├── conf_1/
    |   │   ├── output.cif # or output.pdb
    |   ├── ...
    |── mmcif_id_2/
    |   ├── conf_0/
    |   │   ├── output.cif # or output.pdb
    |   ├── ...
    |── ...
    """
    metrics_path = _run_recent_pdb_eval_benchmark(
        dataset_folder=dataset_folder,
        dataset_index_path=dataset_index_path,
        predictions_folder=predictions_folder,
        num_conf=num_conf,
        conf_idx=conf_idx,
        score_top_ranked=score_top_ranked,
        use_cache=use_cache,
    )
    typer.echo(f"Metrics saved to {metrics_path}")


def _run_posebusters_benchmark(
    dataset_folder: str,
    predictions_folder: str,
    target_ids: list[str] | None = None,
    num_conf: int | None = None,
    conf_idx: int | str | None = None,
    score_top_ranked: bool = False,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    use_cache: bool = False,
) -> Path:
    logger = LOGGER

    if target_ids is None:
        target_ids = get_benchmark_target_ids(dataset_folder)

    logger.info(f"Computing posebusters metrics for {len(target_ids)} target ids")

    if score_top_ranked:
        if conf_idx is not None:
            raise ValueError("Cannot specify conf_idx when scoring top ranked conformations.")
        # Convention: the top ranked conformation is the one with the highest LG1 score
        conf_ids = ["best_LG1"]
    else:
        if conf_idx is None:
            if num_conf is None:
                num_conf = get_num_conformers(predictions_folder, target_ids)
            conf_ids = [f"conf_{i}" for i in range(num_conf)]
        else:
            if num_conf is not None:
                raise ValueError("Cannot specify conf_idx when num_conf is provided.")
            conf_ids = [f"conf_{conf_idx}"]

    metrics_per_conf: dict[int | str, list[pd.DataFrame]] = defaultdict(list)
    failures_per_conf: dict[int | str, list[pd.DataFrame]] = defaultdict(list)
    for i, target_id in enumerate(target_ids):
        logger.info(f"Processing target id {target_id} ({i + 1}/{len(target_ids)})")

        this_results = compute_posebusters_metrics_for_target_id(
            target_id=target_id,
            dataset_folder=dataset_folder,
            predictions_folder=predictions_folder,
            conf_names=conf_ids,
            pocket_cutoff=pocket_cutoff,
            use_cache=use_cache,
            logger=logger,
        )
        for conf_idx, (metrics_df, failure_df) in zip(conf_ids, this_results):
            metrics_per_conf[conf_idx].append(metrics_df)
            failures_per_conf[conf_idx].append(failure_df)

    metrics_dfs = {conf_idx: pd.concat(metrics_per_conf.get(conf_idx, [pd.DataFrame()])) for conf_idx in conf_ids}
    failures_dfs = {conf_idx: pd.concat(failures_per_conf.get(conf_idx, [pd.DataFrame()])) for conf_idx in conf_ids}

    # save the metrics and failures to csv
    metrics_folder = Path(predictions_folder) / "metrics"
    metrics_folder.mkdir(exist_ok=True)
    for conf_idx in conf_ids:
        metrics_path = f"{metrics_folder}/{conf_idx}_metrics.csv"
        failures_path = f"{metrics_folder}/{conf_idx}_failures.csv"

        metrics_dfs[conf_idx].to_csv(metrics_path, index=False)
        failures_dfs[conf_idx].to_csv(failures_path, index=False)

        # Metric summary
        typer.echo(f"====== Summary of {conf_idx} ===========")
        typer.echo(metrics_dfs[conf_idx].replace({True: 1, False: 0}).describe().T)
        typer.echo("========================================")

    return metrics_folder


def compute_posebusters_metrics_for_target_id(
    target_id: str,
    dataset_folder: str,
    predictions_folder: str,
    conf_names: list[str],
    pocket_cutoff: float,
    use_cache: bool,
    logger: Logger | LoggerAdapter,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    ref_pdb_path = get_ref_protein_pdb_path(dataset_folder, target_id)
    ref_sdf_path = get_ref_ligand_sdf_path(dataset_folder, target_id)

    results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for conf_name in conf_names:
        logger.info(f"Computing metrics for {target_id} conformation {conf_name}")

        conf_folder = get_pred_conf_folder(predictions_folder, target_id, conf_name)
        pred_pdb_path = get_pred_protein_pdb_path(predictions_folder, target_id, conf_name)
        pred_sdf_path = get_pred_ligand_sdf_path(predictions_folder, target_id, conf_name)

        if not Path(pred_pdb_path).is_file() or not Path(pred_sdf_path).is_file():
            logger.warning(
                f"Could not find predicted pdb or sdf file for {target_id} conformation {conf_name}. Skipping."
            )
            failures_df = pd.DataFrame(
                {"target_id": [target_id], "error_msg": [f"Could not find predicted pdb or sdf file"]}
            )
            results.append((pd.DataFrame(), failures_df))
            continue

        if use_cache:
            cached_results = get_cached_results(conf_folder)
            if cached_results is not None:
                logger.info(f"Found existing metrics/failures for {target_id} conformation {conf_name}. Loading.")
                results.append(cached_results)
                continue

        metrics_df, failures_df = compute_metrics(
            target_id=target_id,
            ref_protein_pdb=Path(ref_pdb_path),
            ref_ligand_sdf=Path(ref_sdf_path),
            pred_protein_pdb=Path(pred_pdb_path),
            pred_ligand_sdf=Path(pred_sdf_path),
            output_dir=Path(conf_folder),
            pocket_cutoff=pocket_cutoff,
            logger=logger,
            return_exceptions=True,
        )

        # save the metrics and failures to csv
        metrics_df.to_csv(get_conformer_metrics_path(conf_folder), index=False)
        failures_df.to_csv(get_conformer_failure_path(conf_folder), index=False)

        results.append((metrics_df, failures_df))

    return results


def get_benchmark_target_ids(dataset_folder: str) -> list[str]:
    target_ids = []
    for p in Path(dataset_folder).iterdir():
        if p.is_dir():
            target_id = p.name
            protein_pdb_path = p / f"{target_id}_protein.pdb"
            if protein_pdb_path.is_file():
                target_ids.append(target_id)

    return sorted(target_ids)


def _run_recent_pdb_eval_benchmark(
    dataset_folder: str,
    dataset_index_path: str,
    predictions_folder: str,
    mmcif_ids: list[str] | None = None,
    num_conf: int | None = None,
    conf_idx: int | str | None = None,
    score_top_ranked: bool = False,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    use_cache: bool = False,
) -> Path:
    logger = LOGGER

    # save the metrics and failures to csv
    metrics_folder = Path(predictions_folder) / "metrics"
    metrics_folder.mkdir(exist_ok=True)

    if mmcif_ids is not None and len(mmcif_ids) == 0:
        logger.info("No mmcif ids provided. Skipping the computation.")
        return metrics_folder

    dataset_index_df = pd.read_csv(dataset_index_path)

    if mmcif_ids is None:
        mmcif_ids = get_all_mmcif_ids_from_dataset_index_df(dataset_index_df)

    if conf_idx is None:
        if num_conf is None:
            num_conf = get_num_conformers(predictions_folder, mmcif_ids)
        conf_ids = [f"conf_{i}" for i in range(num_conf)]
    else:
        if score_top_ranked:
            raise ValueError("Cannot specify conf_idx when scoring top ranked conformations.")
        conf_ids = [f"conf_{conf_idx}"]

    if score_top_ranked:
        conf_ids = []

    logger.info(f"Computing internal metrics for {len(mmcif_ids)} mmcif ids.")

    metrics_per_conf: dict[int | str, list[pd.DataFrame]] = defaultdict(list)
    failures_per_conf: dict[int | str, list[pd.DataFrame]] = defaultdict(list)
    # for i, mmcif_id in enumerate(mmcif_ids):
    for i, row in dataset_index_df.iterrows():
        mmcif_id = row["mmcif_id"]
        logger.info(f"Processing mmcif id {mmcif_id} ({i + 1}/{len(dataset_index_df)})")

        chain_or_interface_ids_ = [row["this_chain_or_ccd_id"], row["this_interface_id"]]
        chain_or_interface_ids = [c for c in chain_or_interface_ids_ if pd.notna(c)]

        if score_top_ranked:
            conf_ids = [f"best_{one(chain_or_interface_ids)}"]

        for conf_idx in conf_ids:
            metrics_df, failure_df = compute_internal_metrics_for_mmcif_id(
                mmcif_id=mmcif_id,
                chain_or_interface_ids=chain_or_interface_ids,
                dataset_folder=dataset_folder,
                predictions_folder=predictions_folder,
                conf_name=conf_idx,
                pocket_cutoff=pocket_cutoff,
                use_cache=use_cache,
                logger=logger,
            )
            metrics_per_conf[conf_idx].append(metrics_df)
            failures_per_conf[conf_idx].append(failure_df)

    if score_top_ranked:
        metrics_path = f"{metrics_folder}/top_ranked_metrics.csv"
        failures_path = f"{metrics_folder}/top_ranked_failures.csv"
        metrics_df = pd.concat([pd.concat(metrics_df_) for metrics_df_ in metrics_per_conf.values()])
        failures_df = pd.concat([pd.concat(failures_df_) for failures_df_ in failures_per_conf.values()])
        metrics_df.to_csv(metrics_path, index=False)
        failures_df.to_csv(failures_path, index=False)
        return metrics_folder

    metrics_dfs = {conf_idx: pd.concat(metrics_per_conf.get(conf_idx, [pd.DataFrame()])) for conf_idx in conf_ids}
    failures_dfs = {conf_idx: pd.concat(failures_per_conf.get(conf_idx, [pd.DataFrame()])) for conf_idx in conf_ids}

    for conf_idx in conf_ids:
        metrics_path = f"{metrics_folder}/{conf_idx}_metrics.csv"
        failures_path = f"{metrics_folder}/{conf_idx}_failures.csv"

        metrics_dfs[conf_idx].to_csv(metrics_path, index=False)
        failures_dfs[conf_idx].to_csv(failures_path, index=False)

    return metrics_folder


def compute_internal_metrics_for_mmcif_id(
    mmcif_id: str,
    chain_or_interface_ids: list[str],
    dataset_folder: str,
    predictions_folder: str,
    conf_name: int | str,
    pocket_cutoff: float,
    use_cache: bool,
    logger: Logger | LoggerAdapter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute internal metrics for a given mmcif_id and its predicted conformations.
    """

    ref_cif_path = get_ref_cif_path(dataset_folder, mmcif_id)
    if not ref_cif_path.is_file():
        logger.error(f"Could not find reference cif file for {mmcif_id}. Skipping.")
        failures_df = pd.DataFrame({"mmcif_id": [mmcif_id], "error_msg": [f"Could not find reference cif file"]})
        return (pd.DataFrame(), failures_df)

    ref_npz_path = maybe_generate_npz_file(ref_cif_path, logger)

    conf_folder = get_pred_conf_folder(predictions_folder, mmcif_id, conf_name)

    if use_cache:
        cached_results = get_cached_results(conf_folder)
        if cached_results is not None:
            logger.info(f"Found existing metrics/failures for {mmcif_id} conformation {conf_name}. Loading.")
            return cached_results

    pred_cif_path = get_pred_cif_or_pdb_path(predictions_folder, mmcif_id, conf_name)
    if not pred_cif_path.is_file():
        logger.error(
            f"Could not find predicted cif or pdb file for {mmcif_id} conformation {conf_name}, path {pred_cif_path}. Skipping."
        )
        failures_df = pd.DataFrame({"mmcif_id": [mmcif_id], "error_msg": [f"Could not find predicted cif file"]})
        return pd.DataFrame(), failures_df

    pred_npz_path = maybe_generate_npz_file(pred_cif_path, logger)

    if not Path(pred_npz_path).is_file():
        logger.warning(f"Could not find predicted npz file for {mmcif_id} conformation {conf_name}. Skipping.")
        failures_df = pd.DataFrame({"mmcif_id": [mmcif_id], "error_msg": [f"Could not find predicted npz file"]})
        return pd.DataFrame(), failures_df

    metrics_df, failures_df = compute_internal_metrics(
        mmcif_id=mmcif_id,
        chain_or_interface_ids=chain_or_interface_ids,
        ref_npz_path=Path(ref_npz_path),
        pred_npz_path=Path(pred_npz_path),
        pocket_cutoff=pocket_cutoff,
        logger=logger,
        return_exceptions=True,
    )
    metrics_df.to_csv(get_conformer_metrics_path(conf_folder), index=False)
    failures_df.to_csv(get_conformer_failure_path(conf_folder), index=False)
    return metrics_df, failures_df


def maybe_generate_npz_file(cif_or_pdb_path: Path, logger: Logger | LoggerAdapter) -> Path:
    npz_path = cif_or_pdb_path.with_suffix(".scoring.npz")
    if npz_path.is_file():
        logger.info(f"Found npz file for {cif_or_pdb_path}.")
    else:
        logger.info(f"Generating npz file from {cif_or_pdb_path}.")
        if cif_or_pdb_path.suffix == ".cif":
            npi = load_mm_cif(str(cif_or_pdb_path), all_residue_template_match=True)
        elif cif_or_pdb_path.suffix == ".pdb":
            npi = load_pdb_and_sdfs(pdb_path=Path(cif_or_pdb_path), parmed_all_residue_template_match=False)
        else:
            raise ValueError(f"Unsupported file format: {cif_or_pdb_path.suffix}")
        dump_nplx_v3_input(npi, npz_path)

    return npz_path


def get_num_conformers(predictions_folder: str, sample_ids: list[str]) -> int:
    sample_id = random.choice(sample_ids)
    sample_folder = Path(predictions_folder) / sample_id
    conformer_folders = [p for p in sample_folder.iterdir() if p.is_dir() and p.name.startswith("conf_")]
    return len(conformer_folders)


def get_cached_results(conf_folder: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    metrics_csv_path = get_conformer_metrics_path(conf_folder)
    failures_csv_path = get_conformer_failure_path(conf_folder)
    if Path(metrics_csv_path).is_file() and Path(failures_csv_path).is_file():
        metrics_df = pd.read_csv(metrics_csv_path)
        failures_df = pd.read_csv(failures_csv_path)
        return metrics_df, failures_df
    else:
        return None


@app.command(name="plot-stats")
def plot_combined_stats_command(
    method_names: list[str] = typer.Option(["AF2-M 2.3", "NP3"], "--method-name", help="List of method names."),
    scoring_dfs: list[str] = typer.Option(
        ["results/af2m_results/metrics/conf_1_metrics.csv", "results/NPv3-base-ranked/metrics/top_ranked_metrics.csv"],
        "--scoring-df",
        help="List of csv files containing the stats.",
    ),
    index_file: str = typer.Option(
        "results/recent_pdb_eval_set_v2_w_CASP15RNA_reduced.csv", "-i", "--index-file", help="Path to the index file."
    ),
) -> None:
    dfs_to_aggregate = {method_name: pd.read_csv(df) for method_name, df in zip(method_names, scoring_dfs)}
    index_file_df = pd.read_csv(index_file)
    plot_combined_stats(dfs_to_aggregate, index_file_df)


if __name__ == "__main__":
    app()

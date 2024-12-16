import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# CASP15 RNAs
CASP15_RNA_LIST = [
    "7PTK_1",
    "7PTL_1",
    "7QR3_1",
    "7QR4_1",
    "7ZJ4_1",
    "8FZA_1",
    "8S95_1",
    "8UYE_1",
    "8UYG_1",
    "8UYJ_1",
    "8UYS_1",
]
CASP15_RNA_LIST_SHORT = [
    "8S95_1",
    "8FZA_1",
    "8BTZ_1",
    "7ZJ4_1",
    "7PTK_1",
    "7PTL_1",
    "7YR6_1",
    "7YR7_1",
    "7QR4_1",
]


def flatten_recentpdbeval_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    df["target_id"] = df["mmcif_id"] + "_" + df["chain_or_interface_id"]
    # Expand the unique "metrics" for the same conformer_id into multiple columns
    flattened_data = {}
    for group in df.groupby("target_id"):
        target_id, group_df = group
        this_group_data = {
            "target_id": group_df["target_id"].iloc[0],
            "mmcif_id": group_df["mmcif_id"].iloc[0],
            "chain_or_interface_id": group_df["chain_or_interface_id"].iloc[0],
        }
        for metric, value in zip(group_df["metrics"], group_df["value"]):
            this_group_data[metric] = value
        flattened_data[target_id] = this_group_data
    df = pd.DataFrame.from_dict(flattened_data, orient="index")
    return df


def annotate_entity_ids(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics_to_keep = [
        "target_id",
        "entity",
        "conformer_id",
        "target_entity_id",
        # Default ligand metrics
        "tm_score",
        "ligand_rmsd",
        "pocket_rmsd",
        "protein_tm_score",
        "protein_rmsd",
        "sanitization",
        "inchi_convertible",
        "all_atoms_connected",
        "molecular_formula",
        "molecular_bonds",
        "double_bond_stereochemistry",
        "tetrahedral_chirality",
        "bond_lengths",
        "bond_angles",
        "internal_steric_clash",
        "aromatic_ring_flatness",
        "double_bond_flatness",
        "internal_energy",
        "protein-ligand_maximum_distance",
        "minimum_distance_to_protein",
        "volume_overlap_with_protein",
        "minimum_distance_to_organic_cofactors",
        # Default chain/interface metrics
        "DockQ_≥_0.23",
        "DockQ",
        "rmsd_≤_2å",
    ]
    if "chain_or_interface_id" in metrics_df.columns:
        metrics_df = flatten_recentpdbeval_metrics_df(metrics_df)
        metrics_df["target_id"] = metrics_df["mmcif_id"]
        metrics_df["entity"] = metrics_df["chain_or_interface_id"].replace({"global": "overall"})
    metrics_df["target_entity_id"] = metrics_df["target_id"].str.upper() + "_" + metrics_df["entity"].astype(str)
    metrics_df = metrics_df[metrics_df.columns.intersection(metrics_to_keep)]
    return metrics_df


def prepare_heatmap_data(final_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create a combined identifier for each model and task
    final_stats["Task_Model"] = final_stats.apply(
        lambda row: f"{row['Task']} - {row['DataFrame_Name']} (n={int(row['n'])})", axis=1
    )
    # Pivot the DataFrame to have metrics as columns and Task_Model as index
    heatmap_data = final_stats.pivot(index="Task_Model", columns="Metric", values="mean")

    # Prepare the annotations DataFrame
    annotations = final_stats.copy()
    # Format the annotations to include mean and CI
    annotations["annotation"] = annotations.apply(
        lambda row: f"{row['mean']:.3f}\n({row['ci_lower']:.3f}-{row['ci_upper']:.3f})"
        if not np.isnan(row["mean"])
        else "",
        axis=1,
    )
    # Pivot the annotations to match heatmap_data
    annotations = annotations.pivot(index="Task_Model", columns="Metric", values="annotation")

    return heatmap_data, annotations


def mean_sem_per_endpoint_from_raw_metrics(
    dfs: dict[str, pd.DataFrame], validity_metrics: bool = False
) -> pd.DataFrame:
    """
    Combines statistical analysis and heatmap plotting for a given dictionary of DataFrames.

    Args:
    - dfs (dict): A dictionary where keys are DataFrame names and values are DataFrames.
    - save_name (str): The base name to use when saving the CSV and PNG files.

    Returns:
    - combined_stats (DataFrame): A DataFrame with mean statistics for the specified metrics.
    """
    all_stats = []
    # Calculate mean for the specified columns
    metrics = [
        "tm_score",
        "DockQ",
        "DockQ_≥_0.23",
        "DockQ_≥_0.80",
        "rmsd_≤_1å",
        "rmsd_≤_2å",
        "rmsd_≤_5å",
    ]
    if validity_metrics:
        # Posebusters validity metrics
        metrics += [
            "all_atoms_connected",
            "molecular_bonds",
            "double_bond_stereochemistry",
            "tetrahedral_chirality",
            "bond_lengths",
            "bond_angles",
            "internal_steric_clash",
            "aromatic_ring_flatness",
            "double_bond_flatness",
            "internal_energy",
            "protein-ligand_maximum_distance",
        ]

    # Loop through each DataFrame and calculate stats
    for df_name, df in dfs.items():
        if "ligand_rmsd" in df.columns:
            # Calculate conditions for RMSD thresholds
            df["rmsd_≤_1å"] = np.where(np.isnan(df["ligand_rmsd"]), np.nan, df["ligand_rmsd"] <= 1)
            df["rmsd_≤_2å"] = np.where(np.isnan(df["ligand_rmsd"]), np.nan, df["ligand_rmsd"] <= 2)
            df["rmsd_≤_5å"] = np.where(np.isnan(df["ligand_rmsd"]), np.nan, df["ligand_rmsd"] <= 5)
        else:
            # Fill with NaNs if the column is not present
            df["ligand_rmsd"] = np.nan
            df["rmsd_≤_1å"] = np.nan
            df["rmsd_≤_2å"] = np.nan
            df["rmsd_≤_5å"] = np.nan
        if "DockQ" in df.columns:
            # RecentPDBEval metrics
            df["DockQ_≥_0.23"] = np.where(np.isnan(df["DockQ"]), np.nan, df["DockQ"] >= 0.23)
            df["DockQ_≥_0.80"] = np.where(np.isnan(df["DockQ"]), np.nan, df["DockQ"] >= 0.80)
        else:
            # Fill with NaNs if the column is not present
            df["DockQ"] = np.nan
            df["DockQ_≥_0.23"] = np.nan
            df["DockQ_≥_0.80"] = np.nan

        # Initialize a dictionary to store stats
        stats_dict = {}

        for metric in metrics:
            if metric not in df.columns:
                stats_dict[metric] = {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                continue
            data = df[metric].dropna().to_numpy()
            n = len(data)
            if n > 1:
                mean = data.mean()
                sem = stats.sem(data)  # Standard error of the mean
                # Compute 95% confidence interval
                h = sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
                ci_lower = mean - h
                ci_upper = mean + h
                stats_dict[metric] = {"mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper}
            elif n == 1:
                mean = data[0]
                stats_dict[metric] = {"mean": mean, "ci_lower": np.nan, "ci_upper": np.nan}
            else:
                stats_dict[metric] = {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

        # Convert stats_dict to DataFrame
        stats_df = pd.DataFrame(stats_dict)[metrics].transpose()
        stats_df["DataFrame_Name"] = df_name

        # Add task information
        num_samples = df.shape[0]
        stats_df["n"] = num_samples

        # Reset index to have 'Metric' as a column
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={"index": "Metric"}, inplace=True)

        # Add to the list
        all_stats.append(stats_df)

    # Combine all stats into a single DataFrame
    combined_stats = pd.concat(all_stats, axis=0, ignore_index=True)
    return combined_stats


# Plot ranked stats with mmcif inclusion list
def generate_ranked_stats_with_mmcif_inclusion_list(
    dfs: dict[str, pd.DataFrame], inclusion_list: list[str]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    # Filter the DataFrames by the inclusion list
    dfs_filtered = {}
    for key, df in dfs.items():
        dfs_filtered[key] = df[df["target_id"].isin(inclusion_list)].copy()
    return dfs_filtered, mean_sem_per_endpoint_from_raw_metrics(dfs_filtered)


def generate_ranked_stats_with_interface_inclusion_list(
    dfs: dict[str, pd.DataFrame], inclusion_list: list[str]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    # Filter the DataFrames by the inclusion list
    dfs_filtered = {}
    for key, df in dfs.items():
        dfs_filtered[key] = df[df["target_entity_id"].isin(inclusion_list)].copy()
    return dfs_filtered, mean_sem_per_endpoint_from_raw_metrics(dfs_filtered)


def plot_combined_stats(dfs_to_aggregate: dict[str, pd.DataFrame], recentpdb_v2_index: pd.DataFrame) -> None:
    dfs_to_aggregate = {k: annotate_entity_ids(v) for k, v in dfs_to_aggregate.items()}

    # Define target lists
    n_targets = len(recentpdb_v2_index.chain_or_interface_id.unique())
    all_targets = recentpdb_v2_index["chain_or_interface_id"]
    protein_monomer_targets = recentpdb_v2_index[recentpdb_v2_index["eval_type"] == "protein"]["chain_or_interface_id"]
    rna_monomer_targets = recentpdb_v2_index[recentpdb_v2_index["eval_type"] == "RNA"]["chain_or_interface_id"]
    protein_protein_targets = recentpdb_v2_index[
        recentpdb_v2_index["eval_type"].isin(["protein:protein", "peptide:protein"])
    ]["chain_or_interface_id"]
    protein_dna_targets = recentpdb_v2_index[recentpdb_v2_index["eval_type"] == "DNA:protein"]["chain_or_interface_id"]
    protein_rna_targets = recentpdb_v2_index[recentpdb_v2_index["eval_type"] == "RNA:protein"]["chain_or_interface_id"]
    protein_peptide_targets = recentpdb_v2_index[recentpdb_v2_index["eval_type"] == "peptide:protein"][
        "chain_or_interface_id"
    ]
    # Categories based on mmcif_id
    protein_antibody_targets = recentpdb_v2_index[recentpdb_v2_index["is_antibody"] == True]["mmcif_id"]
    modified_residue_targets = recentpdb_v2_index[recentpdb_v2_index["is_modified_residue"] == True]["mmcif_id"]
    covalent_targets = recentpdb_v2_index[recentpdb_v2_index["is_covalent"] == True]["mmcif_id"]

    casp15_rna_entity_list = recentpdb_v2_index[recentpdb_v2_index["mmcif_id"].isin(CASP15_RNA_LIST)][
        "chain_or_interface_id"
    ]
    casp15_rna_entity_list_short = recentpdb_v2_index[recentpdb_v2_index["mmcif_id"].isin(CASP15_RNA_LIST_SHORT)][
        "chain_or_interface_id"
    ]

    # Define your tasks in a list
    tasks = [
        {"title": "All Targets", "inclusion_list": all_targets, "filter_type": "interface"},
        {"title": "Protein Monomer Targets", "inclusion_list": protein_monomer_targets, "filter_type": "interface"},
        {"title": "RNA Monomer Targets", "inclusion_list": rna_monomer_targets, "filter_type": "interface"},
        {"title": "CASP15 RNA Targets", "inclusion_list": casp15_rna_entity_list, "filter_type": "interface"},
        {
            "title": "CASP15 RNA (AF3 version)",
            "inclusion_list": casp15_rna_entity_list_short,
            "filter_type": "interface",
        },
        {"title": "Protein-Protein Targets", "inclusion_list": protein_protein_targets, "filter_type": "interface"},
        {"title": "Protein-DNA Targets", "inclusion_list": protein_dna_targets, "filter_type": "interface"},
        {"title": "Protein-RNA Targets", "inclusion_list": protein_rna_targets, "filter_type": "interface"},
        {"title": "Protein-Antibody Targets", "inclusion_list": protein_antibody_targets, "filter_type": "mmcif"},
        {"title": "Protein-Peptide Targets", "inclusion_list": protein_peptide_targets, "filter_type": "interface"},
        {"title": "Modified Residue Targets", "inclusion_list": modified_residue_targets, "filter_type": "mmcif"},
        {"title": "Covalent Ligand Targets", "inclusion_list": covalent_targets, "filter_type": "mmcif"},
    ]

    # Collect all stats
    all_combined_stats = []

    for task in tasks:
        # Determine which filtering function to use based on 'filter_type'
        if task["filter_type"] == "interface":
            dfs_filtered, stats = generate_ranked_stats_with_interface_inclusion_list(
                dfs_to_aggregate, task["inclusion_list"]
            )
        elif task["filter_type"] == "mmcif":
            dfs_filtered, stats = generate_ranked_stats_with_mmcif_inclusion_list(
                dfs_to_aggregate, task["inclusion_list"]
            )
        else:
            raise ValueError(f"Unknown filter type: {task['filter_type']}")
        # Add task information
        stats["Task"] = task["title"]
        all_combined_stats.append(stats)

    # Combine all stats into a single DataFrame
    final_stats = pd.concat(all_combined_stats, axis=0, ignore_index=True)

    # Prepare data for heatmap
    heatmap_data, annotations = prepare_heatmap_data(final_stats)

    # Plotting the heatmap
    plt.figure(figsize=(16, len(heatmap_data) * 0.6))
    sns.set(font_scale=0.8)  # Adjust to make the text readable

    ax = sns.heatmap(
        heatmap_data,
        annot=annotations,
        fmt="",
        cmap="RdYlGn",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        center=0.5,
        cbar_kws={"label": "Mean Value"},
    )

    ax.set_title(f"RecentPDBEval Result, N={n_targets}", fontsize=16)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Models", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the DataFrames
    dfs_to_aggregate = {
        "AF2-M 2.3": pd.read_csv("results/af2m_results/metrics/conf_1_metrics.csv"),
        "NP3": pd.read_csv("results/NPv3-base-ranked/metrics/top_ranked_metrics.csv"),
    }
    index_file = pd.read_csv("results/recent_pdb_eval_set_v2_w_CASP15RNA_reduced.csv")

    # Plot the combined stats
    plot_combined_stats(dfs_to_aggregate, index_file)

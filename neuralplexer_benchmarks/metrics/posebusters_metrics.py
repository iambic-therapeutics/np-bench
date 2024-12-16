import io
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired, check_output

import pandas as pd

POSEBUSTERS_TIMEOUT_SECONDS = 120


def run_posebusters_cmd(
    predicted_ligand_sdf_path: Path, reference_ligand_sdf_path: Path, protein_pdb_path: Path
) -> pd.DataFrame:
    cmd = (
        "bust",
        str(predicted_ligand_sdf_path),
        "-l",
        str(reference_ligand_sdf_path),
        "-p",
        str(protein_pdb_path),
        "--outfmt",
        "csv",
    )
    try:
        result = check_output(cmd, shell=False, encoding="UTF-8", timeout=POSEBUSTERS_TIMEOUT_SECONDS)
    except TimeoutExpired:
        raise TimeoutError(f"Posebusters command timed out after {POSEBUSTERS_TIMEOUT_SECONDS} seconds.")
    except CalledProcessError as e:
        raise ValueError(f"Posebusters command failed with the following error: {e.stderr}")

    df = pd.read_csv(io.StringIO(result), sep=",")
    df.drop(columns=["file", "molecule", "mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"], inplace=True)

    return df

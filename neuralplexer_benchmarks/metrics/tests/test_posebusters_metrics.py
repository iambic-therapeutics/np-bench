import pandas as pd

from neuralplexer_benchmarks.metrics.posebusters_metrics import run_posebusters_cmd

EXPECTED_PB_COLUMNS = {
    "sanitization",
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
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
    "rmsd_≤_2å",
}


def test_run_posebusters_cmd(
    protein_pdb_path,
    ligand_sdf_path,
    aligned_ligand_sdf_path,
):
    result = run_posebusters_cmd(
        predicted_ligand_sdf_path=aligned_ligand_sdf_path,
        reference_ligand_sdf_path=ligand_sdf_path,
        protein_pdb_path=protein_pdb_path,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    assert set(result.columns).issuperset(EXPECTED_PB_COLUMNS)

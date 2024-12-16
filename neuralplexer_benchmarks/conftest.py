import asyncio
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "tests" / "data"


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop.set_debug(True)
    yield loop
    loop.close()


@pytest.fixture(name="protein_pdb_path")
def fixture_protein_pdb_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_protein.pdb"


@pytest.fixture(name="protein_fasta_path")
def fixture_protein_fasta_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_protein.fasta"


@pytest.fixture(name="ligand_sdf_path")
def fixture_ligand_sdf_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_ligand.sdf"


@pytest.fixture(name="complex_pdb_path")
def fixture_complex_pdb_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_complex.pdb"


@pytest.fixture(name="predicted_protein_pdb_path")
def fixture_predicted_protein_pdb_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_protein_pred.pdb"


@pytest.fixture(name="predicted_ligand_sdf_path")
def fixture_predicted_ligand_sdf_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_ligand_pred.sdf"


@pytest.fixture(name="aligned_ligand_sdf_path")
def fixture_aligned_ligand_sdf_path() -> Path:
    return TEST_DATA_DIR / "5S8I_2LY_ligand_aligned.sdf"


@pytest.fixture(name="ref_multi_chain_pdb")
def fixture_ref_multi_chain_pdb():
    return TEST_DATA_DIR / "7OKF_VH5_ref.pdb"


@pytest.fixture(name="pred_multi_chain_pdb")
def fixture_pred_multi_chain_pdb():
    return TEST_DATA_DIR / "7OKF_VH5_pred.pdb"


@pytest.fixture(name="ref_multi_chain_npz")
def fixture_ref_multi_chain_npz():
    return TEST_DATA_DIR / "7OKF_VH5_ref.npz"


@pytest.fixture(name="pred_multi_chain_npz")
def fixture_pred_multi_chain_npz():
    return TEST_DATA_DIR / "7OKF_VH5_pred.npz"


@pytest.fixture(name="ref_multi_ligand_npz")
def fixture_ref_multi_ligand_npz():
    return TEST_DATA_DIR / "6VTA_AKN_ref.npz"


@pytest.fixture(name="pred_multi_ligand_npz")
def fixture_pred_multi_ligand_npz():
    return TEST_DATA_DIR / "6VTA_AKN_pred.npz"

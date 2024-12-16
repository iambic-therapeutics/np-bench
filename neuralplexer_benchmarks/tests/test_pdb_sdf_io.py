from pathlib import Path

import pytest
from rdkit import Chem

from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input, load_nplx_v3_input
from neuralplexer_benchmarks.pdb_sdf_io import (
    get_chain_ids_for_ligand_residue_name,
    load_rdkit_mol,
    make_nplx_v3_input,
    rewrite_ligand_sdf_with_reference_sdf,
    rewrite_pdb_with_nplx_v3_input,
)


@pytest.fixture(name="reference_ligand_sdf")
def fixture_reference_ligand_sdf() -> Path:
    return Path(__file__).parent / "data/reference_ligand.sdf"


@pytest.fixture(name="predicted_ligand_sdf")
def fixture_predicted_ligand_sdf() -> Path:
    return Path(__file__).parent / "data/predicted_ligand.sdf"


def test_load_rdkit_mol(reference_ligand_sdf, predicted_ligand_sdf):
    mol, sanitize = load_rdkit_mol(reference_ligand_sdf)
    assert mol is not None
    assert sanitize is True

    mol, sanitize = load_rdkit_mol(predicted_ligand_sdf)
    assert mol is not None
    assert sanitize is False


def test_make_nplx_v3_input(protein_pdb_path: Path) -> None:
    nplx_v3_input = make_nplx_v3_input(protein_pdb_path)
    assert isinstance(nplx_v3_input, NPLXV3Input)


def test_rewrite_pdb_with_nplx_v3_input(protein_pdb_path: Path, tmp_path: Path) -> None:
    new_protein_pdb = tmp_path / "new_protein.pdb"
    rewrite_pdb_with_nplx_v3_input(protein_pdb_path, new_protein_pdb)
    assert new_protein_pdb.exists()


def test_rewrite_ligand_sdf_with_reference_sdf_with_same_atom_ordering(
    ligand_sdf_path: Path, predicted_ligand_sdf_path: Path, tmp_path: Path
) -> None:
    new_ligand_sdf = tmp_path / "new_ligand.sdf"
    rewrite_ligand_sdf_with_reference_sdf(predicted_ligand_sdf_path, ligand_sdf_path, new_ligand_sdf)

    # Check that the atom block is the same as the predicted ligand SDF
    atom_block_start = 4
    atom_block_end = 17
    assert (
        new_ligand_sdf.read_text().splitlines()[atom_block_start:atom_block_end]
        == predicted_ligand_sdf_path.read_text().splitlines()[atom_block_start:atom_block_end]
    )

    # Check that the bond block is the same as the original ligand SDF
    bond_block_start = 17
    bond_block_end = 31
    assert (
        new_ligand_sdf.read_text().splitlines()[bond_block_start:bond_block_end]
        == ligand_sdf_path.read_text().splitlines()[bond_block_start:bond_block_end]
    )


def test_rewrite_ligand_sdf_with_reference_sdf_with_different_atom_ordering(tmp_path: Path) -> None:
    ref_sdf_block = """5SAK_ZRY_A_404
     RDKit          3D

 18 20  0  0  0  0  0  0  0  0999 V2000
   29.2770    3.6300   58.9240 N   0  0  0  0  0  0  0  0  0  0  0  0
   25.9780    4.7010   58.3310 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.6050    7.1750   56.2560 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.0890    5.4240   57.7950 N   0  0  0  0  0  0  0  0  0  0  0  0
   25.0720    8.1330   55.3650 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.9800    3.1620   59.7230 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.5730    3.1800   59.9280 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.7440    3.6770   59.3570 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.5170    6.3050   56.8800 N   0  0  0  0  0  0  0  0  0  0  0  0
   24.6730    2.1650   60.8750 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.9150    1.6610   61.2540 C   0  0  0  0  0  0  0  0  0  0  0  0
   27.0770    2.1540   60.6860 C   0  0  0  0  0  0  0  0  0  0  0  0
   27.3730    4.7160   58.1030 N   0  0  0  0  0  0  0  0  0  0  0  0
   27.9750    3.8370   58.9110 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.1920    9.0180   54.7640 C   0  0  0  0  0  0  0  0  0  0  0  0
   22.8420    8.9340   55.0180 C   0  0  0  0  0  0  0  0  0  0  0  0
   22.3650    7.9570   55.8620 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.2390    7.0890   56.5010 C   0  0  0  0  0  0  0  0  0  0  0  0
  2  4  2  0
  3  5  2  0
  7  8  2  0
  6  8  1  0
  2  8  1  0
  4  9  1  0
  3  9  1  0
  7 10  1  0
 10 11  2  0
  6 12  2  0
 11 12  1  0
  2 13  1  0
 13 14  1  0
  1 14  2  0
  6 14  1  0
  5 15  1  0
 15 16  2  0
 16 17  1  0
 17 18  2  0
  3 18  1  0
M  END
$$$$
"""

    pred_sdf_block = """
     RDKit          2D

 18 20  0  0  0  0  0  0  0  0999 V2000
   -2.6401   -3.1891    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2135   -2.7256    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0000   -3.6073    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2135   -2.7256    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.6401   -3.1891    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.9520   -4.6564    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    4.3786   -5.1199    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.6904   -6.5871    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.1170   -7.0506    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.2317   -6.0469    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9199   -4.5797    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.4933   -4.1162    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  3
  2  3  1  0
  3  4  1  0
  4  5  2  0
  5  6  1  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
 10 11  1  0
 11 12  2  0
  4 13  1  0
 13 14  2  0
 14 15  1  0
 15 16  2  0
 16 17  1  0
 17 18  2  0
 18  2  1  0
 12  7  1  0
 18 13  1  0
M  END
"""

    ref_sdf_path = tmp_path / "ref.sdf"
    ref_sdf_path.write_text(ref_sdf_block)

    pred_sdf_path = tmp_path / "pred.sdf"
    pred_sdf_path.write_text(pred_sdf_block)

    new_pred_sdf = tmp_path / "new_pred.sdf"

    # Check that an error is raised when the atom ordering is different and match_atom_ordering is False
    with pytest.raises(ValueError, match="The pred ligand SDF has different atom ordering than the ref ligand SDF"):
        rewrite_ligand_sdf_with_reference_sdf(pred_sdf_path, ref_sdf_path, new_pred_sdf, match_atom_ordering=False)

    # Check that the ligand SDF is rewritten with the same atom ordering as the reference SDF when match_atom_ordering is True
    rewrite_ligand_sdf_with_reference_sdf(pred_sdf_path, ref_sdf_path, new_pred_sdf, match_atom_ordering=True)

    ref_mol = Chem.MolFromMolBlock(ref_sdf_block)
    new_pred_mol = Chem.MolFromMolFile(str(new_pred_sdf))

    ref_atom_types = [atom.GetSymbol() for atom in ref_mol.GetAtoms()]  # type: ignore[call-arg, unused-ignore]
    new_pred_atom_types = [atom.GetSymbol() for atom in new_pred_mol.GetAtoms()]  # type: ignore[call-arg, unused-ignore]

    assert new_pred_atom_types == ref_atom_types


def test_rewrite_ligand_sdf_with_reference_sdf_with_and_without_sanitization(tmp_path: Path) -> None:
    ref_sdf_block = """5SD5_HWI_A_202
     RDKit          3D

 29 31  0  0  0  0  0  0  0  0999 V2000
    8.8630    3.6890   14.7430 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.4300    0.6470   12.7240 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.6600   -0.6810   15.6910 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.4480   -0.4520   16.5260 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0860    0.2120   15.3630 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0040    0.4570   14.3540 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0980    0.6600   11.8040 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.3530    2.1550   12.9480 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.6310   -0.0820   13.7930 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.3090   -0.0160   14.5370 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.7360   -0.9110   16.7120 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1320   -1.6530   17.9780 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.5140    3.2960   15.7650 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9510    2.9530   16.2230 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.9310    3.9730   15.6920 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.6440    6.1610   15.7480 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.7310    4.7350   14.3150 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.6640    0.9040   11.2840 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.2810    1.3920   12.7220 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5350    2.0640   10.2510 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.8260    5.2180   16.2030 N   0  0  0  0  0  0  0  0  0  0  0  0
    8.4890    7.4020   16.2760 N   0  0  0  0  0  0  0  0  0  0  0  0
    9.5990    5.9680   14.8390 N   0  0  0  0  0  0  0  0  0  0  0  0
   10.6780    4.5390   13.4070 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.6600    1.0880   13.1780 N   0  0  0  0  0  0  0  0  0  0  0  0
    8.9990    2.4080   14.2250 O   0  0  0  0  0  0  0  0  0  0  0  0
    6.2310    0.2450   13.5360 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.5680    2.0240    9.4710 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.4380    2.9600   10.2640 O   0  0  0  0  0  0  0  0  0  0  0  0
  4  5  1  0
  5  6  2  0
  2  8  1  0
  2  9  1  0
  3 10  2  0
  6 10  1  0
  3 11  1  0
  4 11  2  0
 11 12  1  0
 13 14  1  0
 14 15  1  0
  1 15  2  0
  1 17  1  0
  7 18  1  0
 18 19  1  0
 18 20  1  0
 15 21  1  0
 16 21  2  0
 16 22  1  0
 16 23  1  0
 17 23  2  0
 17 24  1  0
  6 25  1  0
  7 25  1  0
 19 25  1  0
  1 26  1  0
  8 26  1  0
  9 27  1  0
 10 27  1  0
 20 28  2  0
 20 29  1  0
M  END
$$$$
"""

    pred_sdf_block = """Created by Neuralplexer v3
     RDKit          3D

 29 31  0  0  0  0  0  0  0  0999 V2000
    0.8836    4.0158   -0.5333 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9514    5.1006    0.5175 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7506    6.4041   -0.0802 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5149    6.6152   -0.6530 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8801    7.8670   -1.1349 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0707    8.1082   -1.7005 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.0646    9.0429   -1.1765 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2208    8.9185   -0.6556 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0305    9.9575   -0.6956 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.6686    7.5400   -0.0657 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.9027    7.3675    0.4162 O   0  0  0  0  0  0  0  0  0  0  0  0
    4.0154    7.0708   -0.3991 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.4879    7.2405    0.3843 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.6605    6.1335    1.3476 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.7243    5.1246    0.4764 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.4648    3.8617    0.9394 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.4637    3.6220    2.3065 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.9851    2.2272    2.7286 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.9096    1.9564    4.2593 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.6297    1.2424    1.8619 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.7586    1.6358    0.4627 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1509    2.8780    0.0425 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.3333    3.1557   -1.4327 N   0  0  0  0  0  0  0  0  0  0  0  0
    6.2240    2.1315   -2.0894 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1061    2.0973   -3.1303 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.7057    2.4882   -4.6024 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.7410    1.5671   -5.5265 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.9531    3.7253   -4.8268 O   0  0  0  0  0  0  0  0  0  0  0  0
    4.1591    3.3106   -2.3723 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  1  0
  5  7  2  0
  7  8  1  0
  8  9  1  0
  8 10  2  0
 10 11  1  0
 11 12  1  0
 12 13  1  0
 13 14  1  0
 14 15  1  0
 15 16  1  0
 16 17  2  0
 17 18  1  0
 18 19  1  0
 18 20  2  0
 20 21  1  0
 21 22  2  0
 22 23  1  0
 23 24  1  0
 24 25  1  0
 25 26  1  0
 26 27  2  0
 26 28  1  0
 25 29  1  0
 10  3  1  0
 22 16  1  0
 29 23  1  0
M  END
"""

    ref_sdf_path = tmp_path / "ref.sdf"
    ref_sdf_path.write_text(ref_sdf_block)

    pred_sdf_path = tmp_path / "pred.sdf"
    pred_sdf_path.write_text(pred_sdf_block)

    new_pred_sdf = tmp_path / "new_pred.sdf"

    # Check that the template match failed when attempt_to_sanitize is False
    with pytest.raises(ValueError, match="Failed to match the atoms in the pred and ref ligand SDFs"):
        rewrite_ligand_sdf_with_reference_sdf(pred_sdf_path, ref_sdf_path, new_pred_sdf, attempt_to_sanitize=False)

    # Check that the ligand SDF is rewritten with the same atom ordering as the reference SDF when match_atom_ordering is True
    rewrite_ligand_sdf_with_reference_sdf(pred_sdf_path, ref_sdf_path, new_pred_sdf, attempt_to_sanitize=True)

    ref_mol = Chem.MolFromMolBlock(ref_sdf_block)
    new_pred_mol = Chem.MolFromMolFile(str(new_pred_sdf))

    ref_atom_types = [atom.GetSymbol() for atom in ref_mol.GetAtoms()]  # type: ignore[call-arg, unused-ignore]
    new_pred_atom_types = [atom.GetSymbol() for atom in new_pred_mol.GetAtoms()]  # type: ignore[call-arg, unused-ignore]

    assert new_pred_atom_types == ref_atom_types


def test_get_chain_ids_for_ligand_residue_name(ref_multi_ligand_npz: Path) -> None:
    nplx_v3_input = load_nplx_v3_input(ref_multi_ligand_npz)

    ligand_residue_name = "ACO"
    chain_ids = get_chain_ids_for_ligand_residue_name(nplx_v3_input, ligand_residue_name)

    assert chain_ids == ["A", "B"]

from functools import cache
from pathlib import Path


@cache
def get_resname_to_smiles_dict() -> dict[str, str]:
    smiles_path = Path(__file__).parent / "resources" / "Components-smiles-stereo-oe.smi"
    resname_to_smiles = {}
    for line in smiles_path.read_text().split("\n"):
        line_split = line.split()
        if len(line_split) < 2:
            continue
        smiles, resname = line_split[0], line_split[1]
        resname_to_smiles[resname] = smiles
    return resname_to_smiles


def get_smiles_for_resname(resname: str) -> str:
    resname_to_smiles = get_resname_to_smiles_dict()
    return resname_to_smiles[resname]

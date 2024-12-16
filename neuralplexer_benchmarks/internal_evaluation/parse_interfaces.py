from dataclasses import dataclass


@dataclass
class ChainAndInterfaces:
    polymer_chain_ids: dict[str, str]
    polymer_polymer_interfaces: dict[tuple[str, str], str]
    ligand_polymer_interfaces: dict[tuple[str, str], str]


def resolve_chain_and_interfaces(chain_or_interface_ids: list[str]) -> ChainAndInterfaces:
    """
    Resolve the chain, polymer-polymer, and ligand-polymer interface ids.

    Args:
        interface_ids (list[str]): List of interface ids.

    Returns:
        ChainAndInterfaces: Chain and interface ids.
    """
    polymer_chain_ids = []
    polymer_polymer_interface_ids = []
    ligand_polymer_interface_ids = []

    for chain_or_interface_id in chain_or_interface_ids:
        if "||" in chain_or_interface_id:
            pair = chain_or_interface_id.split("||")
            if pair[0].startswith("poly") and pair[1].startswith("poly"):
                polymer_polymer_interface_ids.append(chain_or_interface_id)
            if pair[0].startswith("lig") and pair[1].startswith("poly"):
                ligand_polymer_interface_ids.append(chain_or_interface_id)
        else:
            chain_type = chain_or_interface_id.split(":")[0]
            if chain_type == "poly":
                polymer_chain_ids.append(chain_or_interface_id)

    return ChainAndInterfaces(
        polymer_chain_ids=get_chain_id_to_poly_chain_id_mapping(polymer_chain_ids),
        polymer_polymer_interfaces=get_interface_id_mapping(polymer_polymer_interface_ids),
        ligand_polymer_interfaces=get_interface_id_mapping(ligand_polymer_interface_ids),
    )


def get_interface_id_mapping(interface_ids: list[str]) -> dict[tuple[str, str], str]:
    """
    Get the interface id mapping.

    For example, if the interface id is "lig:XYZ||poly:A", the mapping will be: {("XYZ", "A"): "lig:XYZ||poly:A"}
    if the interface id is "poly:A||poly:B", the mapping will be: {("A", "B"): "poly:A||poly:B"}

    Args:
        interface_ids (list[str]): List of interface ids.

    Returns:
        dict[tuple[str, str], str]: Dictionary mapping interface ids.
    """
    interface_id_mapping = {}
    for interface_id in interface_ids:
        pair = interface_id.split("||")
        ligand_or_chain_id = pair[0].split(":")[1]
        chain_id = pair[1].split(":")[1]
        interface_id_mapping[(ligand_or_chain_id, chain_id)] = interface_id

    return interface_id_mapping


def get_chain_id_to_poly_chain_id_mapping(poly_chain_ids: list[str]) -> dict[str, str]:
    """
    Get the chain id mapping.

    For example, if the chain id is "poly:A", the mapping will be: {"A": "poly:A"}

    Args:
        chain_ids (list[str]): List of chain ids.

    Returns:
        dict[str, str]: Dictionary mapping chain ids.
    """
    chain_id_mapping = {}
    for chain_id in poly_chain_ids:
        chain_id_mapping[chain_id.split(":")[1]] = chain_id

    return chain_id_mapping

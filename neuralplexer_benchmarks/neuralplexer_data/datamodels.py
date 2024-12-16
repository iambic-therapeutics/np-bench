from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from pydantic import parse_obj_as
from rdkit import Chem

from neuralplexer_benchmarks.neuralplexer_data.exceptions import (
    BondError,
    ChainIdError,
    CorruptedDataError,
    ResidueIdError,
)
from neuralplexer_benchmarks.str_enum import Enum, StrEnum


# Note: Inheriting from Enum, too, is needed to satisfy mypy; the original enum is a C one.
class BondType(Chem.rdchem.BondType, Enum):
    """
    Extension of rdkit's BondType enum by adding a new bond type for disulfide bonds.
    """

    DISULFIDE = 99  # Choosing a value that is definitely not used by rdkit


class ResidueId(NamedTuple):
    chain: str
    number: int
    insertion_code: str


class SequenceResidue(NamedTuple):
    name: str
    residue_id: ResidueId | None


class PolymerType(StrEnum):
    D_PEPTIDE = "D_PEPTIDE"
    L_PEPTIDE = "L_PEPTIDE"
    DNA = "DNA"
    RNA = "RNA"
    D_SACCHARIDE = "D_SACCHARIDE"
    L_SACCHARIDE = "L_SACCHARIDE"
    OTHER = "OTHER"


POLYMER_TYPE_MAP = {
    "polypeptide(D)": PolymerType.D_PEPTIDE,
    "polypeptide(L)": PolymerType.L_PEPTIDE,
    "polydeoxyribonucleotide": PolymerType.DNA,
    "polyribonucleotide": PolymerType.RNA,
    "polysaccharide(D)": PolymerType.D_SACCHARIDE,
    "polysaccharide(L)": PolymerType.L_SACCHARIDE,
}


class AtomIdentifier(NamedTuple):
    residue_id: ResidueId
    atom_name: str


class ExplicitBond(NamedTuple):
    from_atom: AtomIdentifier
    to_atom: AtomIdentifier
    bond_type: int
    bond_length: float


Sequences = dict[
    str, list[SequenceResidue]
]  # 1D sequences [num_residues_orig] of each biopolymer, with mapping to the original residue index
IsExperimentallyResolvedDict = dict[
    str, np.ndarray[Any, np.dtype[np.bool_]]
]  # [num_residues_orig, 1] for each biopolymer
PolymerTypeDict = dict[str, PolymerType]  # [num_residues_orig, 1] for each biopolymer

Bonds = np.ndarray[
    Any, np.dtype[np.int_]
]  # [num_bonds, 3] (from_atom, to_atom, bond_type), bond_type is expressed in the integer value of the corresponding rdkit bond type


Branch = list[ResidueId]
Ligand = list[ResidueId]


# In comment: Number of entries with the given experiment type on 2024-04-25
class ExperimentTypeEnum(StrEnum):
    ELECTRON_CRYSTALLOGRAPHY = "ELECTRON CRYSTALLOGRAPHY"  # 236
    ELECTRON_MICROSCOPY = "ELECTRON MICROSCOPY"  # 19_825
    EPR = "EPR"  # 8
    FIBER_DIFFRACTION = "FIBER DIFFRACTION"  # 39
    FLUORESCENCE_TRANSFER = "FLUORESCENCE TRANSFER"  # 1
    INFRARED_SPECTROSCOPY = "INFRARED SPECTROSCOPY"  # 4
    NEUTRON_DIFFRACTION = "NEUTRON DIFFRACTION"  # 226
    POWDER_DIFFRACTION = "POWDER DIFFRACTION"  # 21
    SOLID_STATE_NMR = "SOLID-STATE NMR"  # 176
    SOLUTION_NMR = "SOLUTION NMR"  # 14_129
    SOLUTION_SCATTERING = "SOLUTION SCATTERING"  # 84
    X_RAY_DIFFRACTION = "X-RAY DIFFRACTION"  # 184_333


@dataclass
class NPLXV3Input:
    atomic_numbers: np.ndarray[Any, np.dtype[np.int_]]  # element names [N, 1], N is the number of atoms
    atom_types_pdb: np.ndarray[
        Any, np.dtype[np.str_]
    ]  # Atom types as written in the PDB file [N, 1] N is the number of atoms
    atom_coordinates: np.ndarray[Any, np.dtype[np.float_]]  # [N, 3]
    chain_id: np.ndarray[Any, np.dtype[np.str_]]  # [N, 1]
    is_hetatm: np.ndarray[Any, np.dtype[np.bool_]]  # [N, 1]
    residue_id: np.ndarray[Any, np.dtype[np.int_]]  # [N, 1]
    residue_name: np.ndarray[Any, np.dtype[np.str_]]  # [N, 1]
    insertion_code: np.ndarray[Any, np.dtype[np.str_]]  # [N, 1]
    b_factor: np.ndarray[Any, np.dtype[np.float_]]  # [N, 1]; values of 0.0 indicate missing data
    bonds: Bonds
    chain_sequences: Sequences  # Polymer entities with their sequences
    # Whether the coordinate is experimentally resolved or imputed by e.g. PDBFixer. Not ready for use yet
    # Jira: https://iambic.atlassian.net/browse/ML3D-195?atlOrigin=eyJpIjoiYmZlOTExZWZhOWY4NDFmMWIyZWVlMTNjOGMxMjg0NmQiLCJwIjoiaiJ9
    is_experimentally_resolved: IsExperimentallyResolvedDict
    polymer_type: PolymerTypeDict
    nonpolymer_entities: list[ResidueId]
    branch_entities: list[Branch]
    is_solvent: np.ndarray[Any, np.dtype[np.bool_]]  # [N, 1]
    is_ion: np.ndarray[Any, np.dtype[np.bool_]]  # [N, 1]
    is_artifact: np.ndarray[Any, np.dtype[np.bool_]]  # [N, 1]
    three_letter_to_smiles: dict[str, str]  # Full SMILES of each ligand based on ccd
    formal_charges: np.ndarray[Any, np.dtype[np.int_]] | None = (
        None  # [N, 1]; CAVEAT: values of 0 may indicate missing data and data may be wrong
    )
    ligands: list[Ligand] = field(default_factory=list)  # Explicitly marked ligands in the structure

    @property
    def full_residue_ids(self) -> list[ResidueId]:
        return [
            ResidueId(chain_id, residue_id, insertion_code)
            for chain_id, residue_id, insertion_code in zip(self.chain_id, self.residue_id, self.insertion_code)
        ]

    @property
    def atom_identifiers(self) -> list[AtomIdentifier]:
        residue_ids = self.full_residue_ids
        return [
            AtomIdentifier(residue_id, atom_name) for residue_id, atom_name in zip(residue_ids, self.atom_types_pdb)
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NPLXV3Input":
        return cls(**data)

    def maybe_drop_by_mask(
        self, mask: np.ndarray, prob: float = 1.0, rng: np.random.Generator = np.random.default_rng()
    ) -> "NPLXV3Input":
        """
        Drops atoms based on a boolean mask
        """
        if prob < 1.0:
            if rng.random() > prob:
                return self
        return self._drop_by_mask(mask)

    def _drop_by_mask(self, mask: np.ndarray) -> "NPLXV3Input":
        atomic_numbers = self.atomic_numbers[mask]
        atom_types_pdb = self.atom_types_pdb[mask]
        atom_coordinates = self.atom_coordinates[mask]
        chain_id = self.chain_id[mask]
        is_hetatm = self.is_hetatm[mask]
        residue_id = self.residue_id[mask]
        residue_name = self.residue_name[mask]
        insertion_code = self.insertion_code[mask]
        b_factor = self.b_factor[mask]
        is_solvent = self.is_solvent[mask]
        is_ion = self.is_ion[mask]
        is_artifact = self.is_artifact[mask]
        formal_charges = self.formal_charges[mask] if self.formal_charges is not None else None
        # TODO mask for is_experimentally_resolved is not implemented
        is_experimentally_resolved = self.is_experimentally_resolved
        # Reindex bond src/dst idxs based on the masked atoms
        bonds = self.bonds[
            np.logical_and(
                mask[self.bonds[:, 0]],
                mask[self.bonds[:, 1]],
            )
        ]
        updated_atom_indices = np.zeros_like(mask, dtype=np.int_)
        updated_atom_indices[mask] = np.arange(np.sum(mask))
        bonds[:, 0] = updated_atom_indices[bonds[:, 0]]
        bonds[:, 1] = updated_atom_indices[bonds[:, 1]]
        return NPLXV3Input(
            atomic_numbers=atomic_numbers,
            atom_types_pdb=atom_types_pdb,
            atom_coordinates=atom_coordinates,
            chain_id=chain_id,
            is_hetatm=is_hetatm,
            residue_id=residue_id,
            residue_name=residue_name,
            insertion_code=insertion_code,
            b_factor=b_factor,
            bonds=bonds,
            chain_sequences=self.chain_sequences,
            is_experimentally_resolved=is_experimentally_resolved,
            polymer_type=self.polymer_type,
            nonpolymer_entities=self.nonpolymer_entities,
            branch_entities=self.branch_entities,
            is_solvent=is_solvent,
            is_ion=is_ion,
            is_artifact=is_artifact,
            three_letter_to_smiles=self.three_letter_to_smiles,
            formal_charges=formal_charges,
        )


def validate_nplx_v3_input(nplx_v3_input: NPLXV3Input) -> NPLXV3Input:
    atomic_numbers = np.array([np.int_(v) for v in nplx_v3_input.atomic_numbers])
    atom_types_pdb = np.array([np.str_(v) for v in nplx_v3_input.atom_types_pdb])
    atom_coordinates = np.array(
        [[np.float_(v[0]), np.float_(v[1]), np.float_(v[2])] for v in nplx_v3_input.atom_coordinates]
    )
    chain_id = np.array([np.str_(v) for v in nplx_v3_input.chain_id])
    is_hetatm = np.array([np.bool_(v) for v in nplx_v3_input.is_hetatm])
    residue_id = np.array([np.int_(v) for v in nplx_v3_input.residue_id])
    residue_name = np.array([np.str_(v) for v in nplx_v3_input.residue_name])
    insertion_code = np.array([np.str_(v) for v in nplx_v3_input.insertion_code])
    b_factor = np.array([np.float_(v) for v in nplx_v3_input.b_factor])
    bonds = np.array([[np.int_(v[0]), np.int_(v[1]), np.int_(v[2])] for v in nplx_v3_input.bonds])
    is_solvent = np.array([np.bool_(v) for v in nplx_v3_input.is_solvent])
    is_ion = np.array([np.bool_(v) for v in nplx_v3_input.is_ion])
    is_artifact = np.array([np.bool_(v) for v in nplx_v3_input.is_artifact])
    chain_sequences = dict(nplx_v3_input.chain_sequences)
    is_experimentally_resolved = dict(nplx_v3_input.is_experimentally_resolved)
    polymer_type = parse_obj_as(PolymerTypeDict, nplx_v3_input.polymer_type)
    nonpolymer_entities = parse_obj_as(list[ResidueId], nplx_v3_input.nonpolymer_entities)
    branch_entities = parse_obj_as(list[Branch], nplx_v3_input.branch_entities)
    three_letter_to_smiles = dict(nplx_v3_input.three_letter_to_smiles)
    formal_charges = (
        np.array([np.int_(v) for v in nplx_v3_input.formal_charges])
        if nplx_v3_input.formal_charges is not None
        else None
    )
    ligands = nplx_v3_input.ligands or []

    # Ensure uniqueness of residue identifiers across sequences
    residue_identifiers_seen: set[ResidueId] = set()
    for sequence in chain_sequences.values():
        for residue in sequence:
            res_id = residue.residue_id
            if res_id is None:
                continue
            # Chain must not be blank; that case is required for storage
            if not res_id.chain:
                raise ResidueIdError(f"Chain ID must be non-blank: {res_id}")
            if res_id in residue_identifiers_seen:
                raise ResidueIdError(f"Residue identifiers must be unique across sequences. Found duplicate: {res_id}")
            residue_identifiers_seen.add(res_id)

    # Ensure that residue names associated with atoms are consistent with the residue names in the sequences
    residue_identifiers_to_residue_name: dict[ResidueId, str] = {}
    for sequence in chain_sequences.values():
        for residue in sequence:
            res_id = residue.residue_id
            if res_id is not None:
                residue_identifiers_to_residue_name[res_id] = residue.name
    for i in range(len(atomic_numbers)):
        res_id = ResidueId(chain_id[i], residue_id[i], insertion_code[i])
        if (
            res_id in residue_identifiers_to_residue_name
            and residue_name[i] != residue_identifiers_to_residue_name[res_id]
        ):
            raise ResidueIdError(
                f"Residue name {residue_name[i]} at index {i} does not match the residue name {residue_identifiers_to_residue_name[res_id]} associated with the residue identifier {res_id}"
            )

    # Ensure uniqueness of chain IDs across sequences
    chain_ids_seen: set[str] = set()
    for chain_sequence in chain_sequences.values():
        chain_ids_this_sequence = {
            sequence_residue.residue_id.chain for sequence_residue in chain_sequence if sequence_residue.residue_id
        }
        if len(chain_ids_this_sequence) > 1:
            raise ChainIdError(f"Chain IDs {chain_ids_this_sequence} found in the same sequence")
        if chain_ids_this_sequence & chain_ids_seen:
            raise ChainIdError(f"Chain IDs {chain_ids_this_sequence} found in multiple sequences")
        chain_ids_seen.update(chain_ids_this_sequence)

    # Check for duplicate bonds
    if len(bonds) > 0 and len(np.unique(bonds[:, :2], axis=0)) != len(bonds):
        raise BondError("Duplicate bonds detected")

    # Ensure uniqueness of ResidueIDs across ligands
    ligand_residue_ids_seen: set[ResidueId] = set()
    for ligand in ligands:
        for ligand_residue_id in ligand:
            if ligand_residue_id in ligand_residue_ids_seen:
                raise ResidueIdError(
                    f"Residue identifiers must be unique across ligands. Found duplicate: {residue_id}"
                )
            ligand_residue_ids_seen.add(ligand_residue_id)

    return NPLXV3Input(
        atomic_numbers=atomic_numbers,
        atom_types_pdb=atom_types_pdb,
        atom_coordinates=atom_coordinates,
        chain_id=chain_id,
        is_hetatm=is_hetatm,
        residue_id=residue_id,
        residue_name=residue_name,
        insertion_code=insertion_code,
        b_factor=b_factor,
        bonds=bonds,
        chain_sequences=chain_sequences,
        is_experimentally_resolved=is_experimentally_resolved,
        polymer_type=polymer_type,
        nonpolymer_entities=nonpolymer_entities,
        branch_entities=branch_entities,
        is_solvent=is_solvent,
        is_ion=is_ion,
        is_artifact=is_artifact,
        three_letter_to_smiles=three_letter_to_smiles,
        formal_charges=formal_charges,
        ligands=ligands,
    )


def _split_sequences(sequence: np.ndarray, chain_lengths: list[int]) -> list[np.ndarray]:
    return np.split(sequence, np.cumsum(chain_lengths)[:-1])


def load_nplx_v3_input(file_path: Path | BytesIO) -> NPLXV3Input:
    with np.load(file_path) as data:
        chain_lengths = data["chain_lengths"]
        sequence_ids = data["sequence_ids"]
        if len(sequence_ids) != len(chain_lengths):
            raise CorruptedDataError("Length of sequence_ids must be the same as the length of chain_lengths.")
        chain_sequences_names = _split_sequences(data["chain_sequences_names"], chain_lengths)
        chain_sequences_chain_ids = _split_sequences(data["chain_sequences_chain_ids"], chain_lengths)
        chain_sequences_indices = _split_sequences(data["chain_sequences_indices"], chain_lengths)
        chain_sequences_insertion_codes = _split_sequences(data["chain_sequences_insertion_codes"], chain_lengths)
        is_experimentally_resolved_list = _split_sequences(data["is_experimentally_resolved"], chain_lengths)
        chain_sequences = {}
        for sequence_id, names, chain_ids, indices, insertion_codes in zip(
            sequence_ids,
            chain_sequences_names,
            chain_sequences_chain_ids,
            chain_sequences_indices,
            chain_sequences_insertion_codes,
        ):
            chain_sequences[sequence_id] = [
                (
                    SequenceResidue(name, ResidueId(chain_id, index, insertion_code))
                    if chain_id
                    else SequenceResidue(name, None)
                )
                for name, chain_id, index, insertion_code in zip(names, chain_ids, indices, insertion_codes)
            ]
        is_experimentally_resolved = dict(zip(sequence_ids, is_experimentally_resolved_list))
        polymer_type = dict(zip(sequence_ids, [PolymerType(v) for v in data["polymer_type"]]))
        nonpolymer_entities: list[ResidueId] = []
        nonpolymer_entities_raw = str(data["nonpolymer_entities"]).split(";") if data["nonpolymer_entities"] else []
        for residue_id_raw_str in nonpolymer_entities_raw:
            residue_id_raw = residue_id_raw_str.split(",")
            if len(residue_id_raw) <= 2:
                residue_id_raw.append("")
            nonpolymer_entities.append(
                ResidueId(chain=residue_id_raw[0], number=int(residue_id_raw[1]), insertion_code=residue_id_raw[2])
            )
        branch_entities: list[Branch] = []
        branch_entities_raw = str(data["branch_entities"]).split("|") if data["branch_entities"] else []
        for branch_raw in branch_entities_raw:
            if branch_raw == "":
                continue
            branch = []
            for residue_id_raw_str in branch_raw.split(";"):
                residue_id_raw = residue_id_raw_str.split(",")
                if len(residue_id_raw) == 2:
                    residue_id_raw.append("")
                branch.append(
                    ResidueId(chain=residue_id_raw[0], number=int(residue_id_raw[1]), insertion_code=residue_id_raw[2])
                )
            branch_entities.append(branch)
        ligands: list[Ligand] = []
        ligands_raw = str(data["ligands"]).split("|") if data.get("ligands") else []
        for ligand_raw in ligands_raw:
            if ligand_raw == "":
                continue
            ligand = []
            for residue_id_raw_str in ligand_raw.split(";"):
                residue_id_raw = residue_id_raw_str.split(",")
                if len(residue_id_raw) == 2:
                    residue_id_raw.append("")
                ligand.append(
                    ResidueId(chain=residue_id_raw[0], number=int(residue_id_raw[1]), insertion_code=residue_id_raw[2])
                )
            ligands.append(ligand)

        return NPLXV3Input(
            atomic_numbers=data["atomic_numbers"],
            atom_types_pdb=data["atom_types_pdb"],
            atom_coordinates=data["atom_coordinates"],
            chain_id=data["chain_id"],
            is_hetatm=data["is_hetatm"],
            residue_id=data["residue_id"],
            residue_name=data["residue_name"],
            insertion_code=data["insertion_code"],
            b_factor=data["b_factor"],
            bonds=data["bonds"],
            chain_sequences=chain_sequences,
            is_experimentally_resolved=is_experimentally_resolved,
            polymer_type=polymer_type,
            nonpolymer_entities=nonpolymer_entities,
            branch_entities=branch_entities,
            is_solvent=data["is_solvent"],
            is_ion=data["is_ion"],
            is_artifact=data["is_artifact"],
            three_letter_to_smiles=dict(
                zip(data["three_letter_to_smiles_keys"], data["three_letter_to_smiles_values"])
            ),
            formal_charges=data.get("formal_charges"),
            ligands=ligands,
        )


def dump_nplx_v3_input(nplx_v3_input: NPLXV3Input, file_path: Path | BytesIO) -> None:
    # Ensure the same order of lists relating to sequence information
    sequence_ids = list(nplx_v3_input.chain_sequences.keys())

    chain_lengths = [len(nplx_v3_input.chain_sequences[sequence_id]) for sequence_id in sequence_ids]
    for sequence_id, chain_length in zip(sequence_ids, chain_lengths):
        if len(nplx_v3_input.is_experimentally_resolved[sequence_id]) != chain_length:
            raise CorruptedDataError(
                "Length of is_experimentally_resolved must be the same as the length of the corresponding chain sequence."
            )

    chain_sequences_names: list[str] = []
    chain_sequences_chain_ids: list[str] = []
    chain_sequences_indices: list[int] = []
    chain_sequences_insertion_codes: list[str] = []
    for sequence_id in sequence_ids:
        chain_sequence = nplx_v3_input.chain_sequences[sequence_id]
        chain_sequences_names.extend(residue.name for residue in chain_sequence)
        # Note: a blank string is used for chain_id to encode that residue_id is None.
        # NPLXV3Input should have been validated to not contain blank chain_ids.
        chain_sequences_chain_ids.extend(
            residue.residue_id.chain if residue.residue_id else "" for residue in chain_sequence
        )
        # If residue_id is None, the index is ignored.
        chain_sequences_indices.extend(
            residue.residue_id.number if residue.residue_id else 0 for residue in chain_sequence
        )
        # If residue_id is None, the insertion code is ignored.
        chain_sequences_insertion_codes.extend(
            residue.residue_id.insertion_code if residue.residue_id else "" for residue in chain_sequence
        )
    nonpolymer_entities = ";".join(
        ",".join(str(v) for v in residue_id) for residue_id in nplx_v3_input.nonpolymer_entities
    )
    branch_entities = "|".join(
        ";".join(",".join(str(v) for v in residue_id) for residue_id in branch_entity)
        for branch_entity in nplx_v3_input.branch_entities
    )
    ligands = "|".join(
        ";".join(",".join(str(v) for v in residue_id) for residue_id in ligand) for ligand in nplx_v3_input.ligands
    )

    content: dict[str, ArrayLike] = {
        "atomic_numbers": nplx_v3_input.atomic_numbers,
        "atom_types_pdb": nplx_v3_input.atom_types_pdb,
        "atom_coordinates": nplx_v3_input.atom_coordinates,
        "chain_id": nplx_v3_input.chain_id,
        "is_hetatm": nplx_v3_input.is_hetatm,
        "residue_id": nplx_v3_input.residue_id,
        "residue_name": nplx_v3_input.residue_name,
        "insertion_code": nplx_v3_input.insertion_code,
        "b_factor": nplx_v3_input.b_factor,
        "bonds": nplx_v3_input.bonds,
        "chain_lengths": chain_lengths,
        "sequence_ids": sequence_ids,
        "chain_sequences_names": chain_sequences_names,
        "chain_sequences_chain_ids": chain_sequences_chain_ids,
        "chain_sequences_indices": chain_sequences_indices,
        "chain_sequences_insertion_codes": chain_sequences_insertion_codes,
        "is_experimentally_resolved": (
            np.concatenate([nplx_v3_input.is_experimentally_resolved[sequence_id] for sequence_id in sequence_ids])
            if nplx_v3_input.is_experimentally_resolved
            else []
        ),
        "polymer_type": np.array([nplx_v3_input.polymer_type[sequence_id].value for sequence_id in sequence_ids]),
        "nonpolymer_entities": nonpolymer_entities,
        "branch_entities": branch_entities,
        "is_solvent": nplx_v3_input.is_solvent,
        "is_ion": nplx_v3_input.is_ion,
        "is_artifact": nplx_v3_input.is_artifact,
        "three_letter_to_smiles_keys": list(nplx_v3_input.three_letter_to_smiles.keys()),
        "three_letter_to_smiles_values": list(nplx_v3_input.three_letter_to_smiles.values()),
        "ligands": ligands,
    }
    if nplx_v3_input.formal_charges is not None:
        content["formal_charges"] = nplx_v3_input.formal_charges

    np.savez_compressed(
        file_path,
        **content,
    )

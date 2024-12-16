import logging

import numpy as np

from neuralplexer_benchmarks.neuralplexer_data.datamodels import NPLXV3Input

logger = logging.getLogger(__name__)


def drop_all_hydrogen_atoms(
    input: NPLXV3Input, prob: float = 1.0, rng: np.random.Generator = np.random.default_rng()
) -> NPLXV3Input:
    """
    Drops all hydrogen atoms to only keep heavy atoms
    """
    mask = input.atomic_numbers != 1
    return input.maybe_drop_by_mask(mask, prob, rng)


def drop_solvent(
    input: NPLXV3Input, prob: float = 1.0, rng: np.random.Generator = np.random.default_rng()
) -> NPLXV3Input:
    """
    Drops solvent atoms
    """
    mask = np.logical_not(input.is_solvent)
    return input.maybe_drop_by_mask(mask, prob, rng)

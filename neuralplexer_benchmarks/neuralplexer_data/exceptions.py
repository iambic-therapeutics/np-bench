class NPLXV3Error(ValueError):
    """Base class for all neuralplexer v3 errors."""

    pass


class CorruptedDataError(NPLXV3Error):
    """Class for all errors related to corrupted or inconsistent data
    as well as data that does not meet assumptions and expectations."""

    pass


class ChainIdError(CorruptedDataError):
    pass


class ResidueIdError(CorruptedDataError):
    pass


class BondError(NPLXV3Error):
    pass

class NPLXV3InputIOError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"NPLXV3InputIOError: {self.message}"


class PocketAlignedRMSDError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"PocketAlignedRMSDError: {self.message}"


class PosebustersError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"PosebustersError: {self.message}"


class DockQError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"DockQError: {self.message}"


class GeneralizedRMSDError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"GeneralizedRMSDError: {self.message}"


class TMScoreError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"TMScoreError: {self.message}"


class LigandUnboundError(Exception):
    def __init__(self, ligand_residue_name, pocket_cutoff):
        self.ligand_residue_name = ligand_residue_name
        self.pocket_cutoff = pocket_cutoff

    def __str__(self):
        return (
            f"LigandUnboundError: Ligand residue {self.ligand_residue_name} is not bound to any polymer chain "
            f"within pocket cutoff of {self.pocket_cutoff} angstroms."
        )

import logging
import re
from pathlib import Path
from subprocess import check_output


def compute_protein_tm_score_and_rmsd(pdb_file: Path, ref_pdb_file: Path) -> tuple[float, float]:
    """
    Computing TM-score for multimers using USalign
           US-align: universal structure alignment of monomeric and complex proteins
       and nucleic acids

       References to cite:
       (1) Chengxin Zhang, Morgan Shine, Anna Marie Pyle, Yang Zhang
           (2022) Nat Methods. 19(9), 1109-1115.
       (2) Chengxin Zhang, Anna Marie Pyle (2022) iScience. 25(10), 105218.

       DISCLAIMER:
         Permission to use, copy, modify, and distribute this program for
         any purpose, with or without fee, is hereby granted, provided that
         the notices on the head, the reference information, and this
         copyright notice appear in all copies or substantial portions of
         the Software. It is provided "as is" without express or implied
         warranty.
    """
    tm_scores = []
    rmsds = []
    try:
        ret = check_output(
            ["USalign", str(pdb_file.absolute()), str(ref_pdb_file.absolute()), "-mm", "1", "-ter", "1"],
            shell=False,
            encoding="UTF-8",
        )
    except Exception:
        logging.error(f"Error while computing TM-score with `auto` mode; trying again with additional '-mol RNA' flag.")
        ret = check_output(
            ["USalign", str(pdb_file.absolute()), str(ref_pdb_file.absolute()), "-mm", "1", "-ter", "1", "-mol", "RNA"],
            shell=False,
            encoding="UTF-8",
        )
    for line in str(ret).split("\n"):
        if re.match(r"^Aligned", line):
            rmsds.append(float(line.split()[4][:-1]))  # Extract the value
        if re.match(r"^TM-score=", line):
            tm_scores.append(float(line.split()[1]))  # Extract the value
    return max(tm_scores), min(rmsds)

"""
Prepare SIC comprehensive data table.

i)   Load and clean table
ii)  Define dropouts
iii) calculate age based on Freesurfer (FS) date, indicating which MRI was pre-processed

!!! note "For demonstration purposes only"
    Note that this script was partially anonymized and simplified for the public repository.
    In general, this script is for demonstration purposes only and not executable in its current form.
    This is due to the restricted access to the LIFE data and the specific server structure.

Author: Simon M. Hofmann | 2021
"""

# %% Import
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np  # noqa: TC002
import pandas as pd

from relevancerelated.utils import cprint

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


# @only_mpi
def load_raw_study_table(exclusion: bool = True, full_table: bool = False, drop_logfile: bool = False) -> pd.DataFrame:
    """
    Load raw study table.

    :param exclusion: dropout SICs with non-usable datasets (MRI)
    :param full_table: whether to delete variables or not (mostly medical information)
    :param drop_logfile: write a logfile about the dropouts
    :return: comprehensive SIC table
    """
    cprint(
        string=f"\nStart loading {'full' if full_table else 'reduced'} MRI table "
        f"{'excluding dropouts' if exclusion else ''}...",
        col="g",
        fm="bo",
    )

    # # # LIFE data Overview
    # # For general information regarding the MRT measurements:

    # Check if @MPI
    p2data = Path(".../data")
    path_to_table = p2data / "Preprocessed/derivatives/"

    if not path_to_table.is_dir():  # due to random access problem
        path_to_table = Path("./TEMP/data_tables/")

    # # # Subset of subjects with MRIs
    table_name = "..._subject_list_inclusion_exclusion....csv"

    print(f"Table name: '{table_name}'")

    mri_table = pd.read_csv(path_to_table / table_name)
    orig_shape = mri_table.shape

    # Write Logfile about dropouts
    if drop_logfile:
        time_stamp = datetime.now().date().isoformat()
        Path("./TEMP/").mkdir(exist_ok=True)
        log_name = Path(f"./TEMP/{time_stamp}_drop_logfile.txt")
        temp_cnt = 0

        while log_name.exists():
            temp_cnt += 1
            log_name = Path(f"{str(log_name).split('file')[0]}file_{temp_cnt}.txt")

        with log_name.open("w") as drop_log:
            drop_log.write(
                "Dropout logfile\n{}\n\nTable: {}\n\nOriginal shape: {}\n{}".format(
                    str(datetime.now()).split(".")[0],
                    table_name,
                    orig_shape,
                    "\n" if exclusion else "No dropouts selected",
                )
            )

    if not full_table:
        # Remove those without an MRI scan:
        have_mri = mri_table["MR_head_y_n"].astype("str") == "yes"  # boolean vec
        mri_table = mri_table[have_mri]

        if drop_logfile:
            drop_log.write(f"post 'have_mri': {mri_table.shape}\t[drop: -{orig_shape[0] - mri_table.shape[0]}]\n")

        #  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

        # Remove irrelevant variables
        rm_col = [...]

        # ...

        mri_table = mri_table.drop(rm_col, axis=1)
        del rm_col, have_mri

    # ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # # # Create list of dropouts
    # # Remove unknown/empty datasets
    if exclusion:
        temp_shape = mri_table.shape
        mri_table = mri_table.drop(mri_table.loc[mri_table["SIC"] == "..."].index)
        # ...

        if drop_logfile:
            drop_log.write(f"post 'single SICs': {mri_table.shape}\t[drop: -{temp_shape[0] - mri_table.shape[0]}]\n")

    # ...

    return mri_table


def sic_info(inp_table: pd.DataFrame, inp_sic: str | list[str] | np.ndarray[str], verbose: bool = False) -> None:
    """
    Provide information about SIC(s) of interest.

    :param inp_table: place mri_table here
    :param inp_sic: string or list of strings
    :param verbose: verbose or not
    """
    mri_overview_vars = ["SIC", "AGE_FS", "Freesurfer_Datum", "SIC_comment"]

    inp_sic = inp_sic if isinstance(inp_sic, list) else [inp_sic]
    for ssic in inp_sic:
        print(inp_table.loc[inp_table["SIC"] == ssic][mri_overview_vars[:-1]])
        print("SIC_comment:", inp_table.loc[inp_table["SIC"] == ssic][mri_overview_vars[-1]].item())
        if verbose:
            print("MR_...:", inp_table.loc[inp_table["SIC"] == ssic]["MR_..."].item())
            print(inp_table.loc[inp_table["SIC"] == ssic][["MRT_....1", "MRT_....2"]])
            print(inp_table.loc[inp_table["SIC"] == ssic][["freesurf_check_result", "corrected_y_n"]])
        print("\n")


# >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END

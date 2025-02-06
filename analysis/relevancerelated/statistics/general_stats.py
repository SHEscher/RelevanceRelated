#!/usr/bin/env python3
"""
General stats for RelevanceRelated.

Author: Simon M. Hofmann
Years: 2024
"""

# %% Import
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from relevancerelated.configs import params, paths
from relevancerelated.modeling.LRP.relevancerelated import (
    ALL_FLAIR_SUBENS_NAMES,
    ALL_T1_SUBENS_NAMES,
)
from relevancerelated.utils import cprint

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
REL_COL_NAME: str = "missing_{seq}_relevance_map"
VERBOSE_SICS_WO_REL_MAPS: bool = False

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def cohen_d(d1: np.ndarray, d2: np.ndarray) -> float:
    """Calculate Cohen's d for independent samples."""
    # calculate the pooled standard deviation
    s = np.sqrt(((len(d1) - 1) * np.var(d1, ddof=1) + (len(d2) - 1) * np.var(d2, ddof=1)) / (len(d1) + len(d2) - 2))
    return (np.mean(d1) - np.mean(d2)) / s  # effect size


def sic_has_relevance_map(sic: str, subens_seq: str) -> bool:
    """
    Check if a SIC has a relevance map for the given sub-ensemble.

    :param sic: SIC to check.
    :param subens_seq: Sub-ensemble sequence to check ['t1', 'flair'].
    :return: True if a relevance map exists, False otherwise.
    """
    # Check arguments
    if not sic.startswith("..."):  # anonymized
        msg = f"Invalid SIC '{sic}'."
        raise ValueError(msg)
    subens_seq = subens_seq.lower()
    if subens_seq not in {"t1", "flair"}:
        msg = f"Unknown sub-ensemble sequence '{subens_seq}'."
        raise ValueError(msg)
    all_subens_names = ALL_T1_SUBENS_NAMES if subens_seq == "t1" else ALL_FLAIR_SUBENS_NAMES

    # Check if SIC has a relevance map for the given sub-ensemble type (T1 or FLAIR)
    has_relevance_map = False
    for subens_name in all_subens_names:
        if list(Path(paths.statsmaps.LRP, subens_name, "aggregated", sic).glob(f"{params.analyzer_type}_*.*")):
            has_relevance_map = True
            break
    return has_relevance_map


def update_overview_table_with_relevance_map_info(rerun: bool = False, verbose: bool = False) -> None:
    """Include overview of existing relevance maps for each SIC."""
    # Load table
    rel_overview_tab = pd.read_csv(paths.data.tables.overview)

    # Fill the table with information on missing relevance maps
    for seq in ["t1", "flair"]:
        tab_col_name = REL_COL_NAME.format(seq=seq)

        if tab_col_name in rel_overview_tab.columns and not rerun:
            if verbose:
                cprint(string=f"{tab_col_name} already present", col="g")
            return

        if tab_col_name in rel_overview_tab.columns and rerun and verbose:
            cprint(string=f"{tab_col_name} already present but will refill the table", col="y")

        for sic in tqdm(rel_overview_tab.SIC_FS, desc=f"Filling '{tab_col_name}' column", total=len(rel_overview_tab)):
            missing_rel_map = not sic_has_relevance_map(sic=sic, subens_seq=seq)
            rel_overview_tab.loc[sic == rel_overview_tab.SIC_FS, tab_col_name] = int(missing_rel_map)

    # Save table
    if verbose:
        cprint(string="Saving updated overview table", col="g")
    rel_overview_tab.to_csv(paths.data.tables.overview, index=False)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


if __name__ == "__main__":
    # Number of subjects in sub-analyses
    cprint(string="Number of subjects in sub-analyses\n", fm="ul")

    # Number of subjects with T1 and FLAIR relevance maps
    update_overview_table_with_relevance_map_info(rerun=False, verbose=True)
    overview_table = pd.read_csv(paths.data.tables.overview, index_col=0)
    for mri_seq in ("t1", "flair"):
        col_name = REL_COL_NAME.format(seq=mri_seq)
        print(
            f"Number of {len(overview_table)} participants with",
            *col_name.split("_"),
            f":\n{overview_table[col_name].value_counts()}\n",
        )

    # Check the overlap of participants without relevance maps of both MRI sequences (T1, FLAIR)
    sics_wo_t1_rel_maps = overview_table.loc[overview_table.missing_t1_relevance_map == 1]
    sics_wo_flair_rel_maps = overview_table.loc[overview_table.missing_flair_relevance_map == 1]
    sics_wo_rel_maps = {
        "t1": sics_wo_t1_rel_maps,
        "flair": sics_wo_flair_rel_maps,
    }
    print(
        f"{len(set(sics_wo_t1_rel_maps.SIC_FS).intersection(sics_wo_flair_rel_maps.SIC_FS))} participants have "
        f"neither T1 nor FLAIR relevance maps.\n"
    )

    if VERBOSE_SICS_WO_REL_MAPS:
        # Get predictions and check in which subsets these participants were
        from relevancerelated.modeling.MRInet.trained import TrainedEnsembleModel

        for mri_seq, all_subens_names in zip(
            ["t1", "flair"], [ALL_T1_SUBENS_NAMES, ALL_FLAIR_SUBENS_NAMES], strict=True
        ):
            print(f"\n{mri_seq.upper()}" + " *-*" * 10)
            for sub_ens_name in all_subens_names:
                print()
                subens_model = TrainedEnsembleModel(model_name=sub_ens_name)
                found_sics = False
                for subset_name in subens_model.get_sics():
                    has_sics = set(subens_model.get_sics()[subset_name]).intersection(sics_wo_rel_maps[mri_seq].SIC_FS)
                    if has_sics:
                        cprint(
                            string=f"{sub_ens_name} has {len(has_sics)} {mri_seq.upper()} participants w/o rel-maps "
                            f"in the {subset_name} subset.",
                            col="g" if subset_name == "test" else "b" if subset_name == "validation" else "y",
                        )
                        if subset_name != "train":
                            print(f"{subset_name}:", has_sics)
                        found_sics = True
                if not found_sics:
                    cprint(string=f"{sub_ens_name} has no {mri_seq.upper()} participants w/o rel-maps.", col="r")

                # Check predictions
                found_sics = False
                for subset_name in ["validation", "test"]:
                    sub_ens_preds, sub_ens_y, sub_ens_pred_sics = subens_model.get_headmodel_data(
                        multi_level=False,
                        subset=subset_name,
                        return_sics=True,
                        verbose=False,
                        dropnan=False,
                    )
                    has_sics = set(sub_ens_pred_sics).intersection(sics_wo_rel_maps[mri_seq].SIC_FS)
                    if has_sics:
                        cprint(
                            string=f"{sub_ens_name} has {mri_seq.upper()} predictions for {len(has_sics)} "
                            f"participants w/o rel-maps in the {subset_name} subset.",
                            col="g" if subset_name == "test" else "b" if subset_name == "validation" else "y",
                        )
                        print(f"{subset_name}:", has_sics)
                        found_sics = True
                    if not found_sics:
                        cprint(
                            string=f"{sub_ens_name} has no {mri_seq.upper()} {subset_name}-set predictions for "
                            f"participants w/o rel-maps.",
                            col="r",
                        )

    # Calculate the length of the longest string
    max_length = max(
        *(len(p.stem.removeprefix("merged_")) for p in Path(paths.statsmaps.DERIVATIVES).glob("merged_*.csv")),
        *(len(p.stem.removeprefix("merged_")) for p in Path(paths.results.PVS).glob("*.csv")),
    )

    # Check first for atlas-based measures
    for path_to_measure in Path(paths.statsmaps.DERIVATIVES).glob("merged_*.csv"):
        print(
            f"N_{path_to_measure.stem.removeprefix('merged_').ljust(max_length)} = "
            f"{pd.read_csv(path_to_measure, index_col=0).subject.nunique()}"
        )
    # Differences are here based on maps from sub-ensembles (T1, or FLAIR), FA, and CS, and atlases

    # Number of subjects for PVS analysis
    print("-" * (max_length + len("   = xxxx")))
    for path_to_pvs in Path(paths.results.PVS).glob("*.csv"):
        print(f"N_{path_to_pvs.stem.ljust(max_length)} = {pd.read_csv(path_to_pvs).SIC.nunique()}")
    # Differences are here based on PVS model (T1, or multi-modal T1-FLAIR) and maps from sub-ensembles (T1, or FLAIR)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

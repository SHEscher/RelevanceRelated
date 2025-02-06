#!/usr/bin/env python3
"""
Run stats on the MRI features.

Author:  Simon M. Hofmann | Ole Goltermann
Years: 2022
"""

# %% Import
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from relevancerelated.configs import params, paths
from relevancerelated.dataloader.atlases import possible_atlases
from relevancerelated.utils import cprint

# %% Set global vars & paths  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

PLOT_ALL: bool = False
SAVE_ALL: bool = False  # whether to save plots
CORRECTION_POLY_DEGREE: int = 3

# Variables of interest
GM_index = ["gray_matter_volume_mm^3", "average_thickness_mm"]
voi_in_cs_tab = ["sum_relevance", "mean_relevance", "brain_age", "age"]  # variables of interest

# %% Functions  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def correct_data_for(
    var2cor: str, correct_for: str, data: pd.DataFrame, deg: int = 3, plot: bool = False
) -> pd.DataFrame:
    """Correct data for given variable."""
    # Dropnan
    _data = data.copy()
    df_sub = _data[[var2cor, correct_for]].copy().dropna()  # temp sub df

    # Extract numpy arrays
    _var2cor = df_sub[var2cor].to_numpy()  # [:, np.newaxis]
    _correct_for = df_sub[correct_for].to_numpy()  # [:, np.newaxis]

    # Fit a polynomial model
    lin_bias_model = np.poly1d(np.polyfit(_correct_for, _var2cor, deg))

    # Correction step
    fit_col = "fit_" + correct_for
    corr_col = f"poly-{deg}-corrected_" + var2cor
    _data[fit_col] = _data[correct_for].map(lin_bias_model)
    _data[corr_col] = _data[var2cor] - _data["fit_" + correct_for]

    # Drop fit column
    _data = _data.drop(columns=[fit_col])

    if plot:
        _fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        sns.regplot(
            x=correct_for,
            y=var2cor,
            data=_data,
            order=deg,
            scatter_kws={"alpha": 0.1, "color": "blue", "fc": "None", "marker": "o", "s": 8},
            line_kws={"color": "red"},
            ax=axs[0],
        )
        sns.regplot(
            x=correct_for,
            y=corr_col,
            data=_data,
            order=3,
            scatter_kws={"alpha": 0.1, "color": "green", "fc": "None", "marker": "o", "s": 8},
            line_kws={"color": "red"},
            ax=axs[1],
        )
        axs[0].set_title(f"{var2cor} vs. {correct_for}")
        axs[1].set_title(f"Corrected {var2cor} vs. {correct_for}")
        _fig.tight_layout()
        plt.show()

    return _data


def process_cs_table(
    cs_table: pd.DataFrame, deg: int = 3, plot: bool = False, age_correct_gm_var: bool = False
) -> pd.DataFrame:
    """
    Process table with cortical surface (CS) data further.

    :param cs_table: CS table (pandas DataFrame)
    :param deg: polynomial degree for correction
    :param plot: whether to plot the corrected variables
    :param age_correct_gm_var: whether to age-correct also the CS/GM related variables
    :return: processed CS table (pandas DataFrame)
    """
    # Compute DBA
    cs_table["DBA"] = cs_table.brain_age - cs_table.age

    # Copmute age-corrected DBA
    cs_table = correct_data_for(var2cor="DBA", correct_for="age", data=cs_table, deg=deg, plot=plot)

    # Compute age-corrected CS variables
    if age_correct_gm_var:
        # this is probably not necessary for the comparison with the age-corrected DBA
        for _gm in GM_index:
            cs_table = correct_data_for(var2cor=_gm, correct_for="age", data=cs_table, deg=deg, plot=plot)

    return cs_table


# %% __main__ o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    from relevancerelated.modeling.LRP.relevancerelated import get_merged_table

    # -------------- Correlation tables and plots per region
    for atl_name in possible_atlases[:2]:
        # Get data for the respective atlas
        cs_tab = get_merged_table(atlas_name=atl_name, feature_abbr="cs" if atl_name not in "aseg" else "subcort")

        # Process CS table (age-correction)
        cs_tab = process_cs_table(cs_table=cs_tab, deg=CORRECTION_POLY_DEGREE, plot=PLOT_ALL, age_correct_gm_var=False)

        # Perform correlation analyses for gray matter volume and cortical thickness
        for gm in GM_index:
            # loop through variables of interest (relevance indices, brain age, age)
            for corr_var_of_choice in voi_in_cs_tab:
                # correlation tables and histogram
                corr_tab = pd.DataFrame(columns=["region", "hemi", "r", "corrected_pval"])
                for structure_name, struc_table in cs_tab.groupby("structure_name"):
                    for hemi, hemi_table in struc_table.groupby("hemisphere"):
                        r, p = pearsonr(hemi_table.dropna()[gm], hemi_table.dropna()[corr_var_of_choice])
                        corr_p = p * (cs_tab.structure_name.nunique() * 2)  # Bonferroni correction
                        corr_tab.loc[len(corr_tab)] = [structure_name, hemi, r, corr_p]
                corr_tab["significant"] = np.where(corr_tab["corrected_pval"] >= params.alpha, "no", "yes")
                corr_tab["direction"] = corr_tab.r.map(lambda x: "positive" if x > 0 else "negative")

                # Print some general info about computed stats
                cprint(string=f"{gm} | {corr_var_of_choice}:", col="b", fm="ul")
                # print(corr_tab.groupby("direction").count().r)  # noqa: ERA001
                print(
                    "R:\n",
                    corr_tab.groupby(["direction", "significant"]).r.describe()[["count", "min", "mean", "max"]],
                )
                # print("P-value:\n",
                #       corr_tab.groupby('direction').corrected_pval.describe()[["count", "min",
                #                                                                "mean", "max"]])
                print("\nSignificant after Bonferroni:")
                print(corr_tab.groupby(["significant", "direction"]).count())
                print("\nStrongest negative correlation:\n", corr_tab[corr_tab.r == corr_tab.r.min()])
                print("\nStrongest positive correlation:\n", corr_tab[corr_tab.r == corr_tab.r.max()])

                if PLOT_ALL:
                    plt.figure(figsize=(10, 10))
                    sns.histplot(
                        corr_tab, x="r", bins=20, hue="significant", hue_order=["yes", "no"], multiple="stack"
                    ).set(title=f"Correlations {corr_var_of_choice} & {gm}")

                    if SAVE_ALL:
                        path_to_plot = Path(paths.results.GM, f"histo_{gm}_{corr_var_of_choice}_{atl_name}.png")
                        plt.savefig(path_to_plot)
                        plt.close()

                        # Save csv file of correlation table
                        path_to_table = Path(paths.results.gm, f"corr_{gm}_{corr_var_of_choice}_{atl_name}.csv")
                        corr_tab.to_csv(path_to_table)
                        plt.close()
                    else:
                        plt.show()

                # overview of region-specific correlations
                for h in ["left", "right"]:
                    ncol = 6
                    nrow = ceil(len(corr_tab[(corr_tab["hemi"] == h) & (corr_tab["significant"] == "yes")]) / ncol)
                    ratio = nrow / ncol
                    c = 1
                    if PLOT_ALL:
                        fig = plt.figure(figsize=(30, 30 * ratio))
                        fig.suptitle(f"Correlation: {corr_var_of_choice} & {gm} - {h} hemisphere", fontsize=25)
                        for structure_name, struc_table in cs_tab.groupby("structure_name"):
                            struc_table = struc_table[struc_table["hemisphere"] == h]  # noqa: PLW2901
                            r, p = pearsonr(struc_table.dropna()[gm], struc_table.dropna()[corr_var_of_choice])
                            corr_p = p * (len(cs_tab.structure_name.unique()) * 2)  # Bonferroni correction
                            if corr_p < params.alpha:
                                plt.subplot(nrow, ncol, c)
                                sns.regplot(x=corr_var_of_choice, y=gm, data=struc_table).set(
                                    title=f"{structure_name} | r={r:.3f} | p={corr_p:.2g}"
                                )
                                c += 1
                        fig.tight_layout()
                        fig.subplots_adjust(top=0.96)

                        if SAVE_ALL:
                            path_to_plot = Path(paths.results.GM, f"corr_{gm}_{corr_var_of_choice}_{atl_name}.png")
                            plt.savefig(path_to_plot)
                            plt.close()
                        else:
                            plt.show()

        # -------------- Plot correlations for the whole brain

        # Plot correlation between variable of interest and gray_matter_volume_mm^3 for the whole brain
        cs_tab["GMV"] = cs_tab["gray_matter_volume_mm^3"]
        glob_gmv = cs_tab.groupby("subject").sum().GMV
        sum_rel = cs_tab.groupby("subject").sum().sum_relevance
        mean_rel = cs_tab.groupby("subject").mean().mean_relevance
        brain_age = cs_tab.groupby("subject").mean().brain_age
        age = cs_tab.groupby("subject").mean().age
        temp_df = pd.DataFrame({
            "relative_global_GMV": glob_gmv,
            "sum_relevance": sum_rel,
            "mean_rel": mean_rel,
            "brain_age": brain_age,
            "age": age,
            "dba": age - brain_age,
        })
        vois = ["sum_relevance", "mean_rel", "brain_age", "age", "dba"]
        for corr_var_of_choice in vois:
            r, p = pearsonr(temp_df.dropna()["relative_global_GMV"], temp_df.dropna()[corr_var_of_choice])
            corr_p = p * (len(cs_tab.structure_name.unique()) + 1)  # Bonferroni correction
            print(
                f"gray_matter_volume_mm^3 ~ {corr_var_of_choice} (whole brain) is "
                f"{'' if corr_p <= params.alpha else 'not '}significant: r={r:.2f}, p={corr_p:.3g}"
            )
            if PLOT_ALL:
                sns.regplot(x=corr_var_of_choice, y="relative_global_GMV", data=temp_df).set(
                    title=f"Whole brain: gray_matter_volume_mm^3 ~ {corr_var_of_choice} | r={r:.3f} | p={p:.2g}"
                )

                if SAVE_ALL:
                    path_to_plot = Path(paths.results.GM, f"corr_gmv_{corr_var_of_choice}_{atl_name}.png")
                    plt.savefig(path_to_plot)
                else:
                    plt.show()

        # Plot correlation between variable of interest and cortical thickness for the whole brain
        num_vertices = cs_tab.groupby("subject").sum().number_of_vertices
        cs_tab = cs_tab.merge(num_vertices, on="subject")
        cs_tab["relative_ct"] = cs_tab["average_thickness_mm"] * (
            cs_tab["number_of_vertices_x"] / cs_tab["number_of_vertices_y"]
        )
        glob_ct = cs_tab.groupby("subject").sum().relative_ct
        sum_rel = cs_tab.groupby("subject").sum().sum_relevance
        mean_rel = cs_tab.groupby("subject").mean().mean_relevance
        brain_age = cs_tab.groupby("subject").mean().brain_age
        age = cs_tab.groupby("subject").mean().age
        temp_df = pd.DataFrame({
            "relative_global_ct": glob_ct,
            "sum_relevance": sum_rel,
            "mean_rel": mean_rel,
            "brain_age": brain_age,
            "age": age,
            "dba": age - brain_age,
        })
        for corr_var_of_choice in vois:
            r, p = pearsonr(temp_df.dropna()["relative_global_ct"], temp_df.dropna()[corr_var_of_choice])
            corr_p = p * (len(cs_tab.structure_name.unique()) + 1)  # Bonferroni correction
            print(
                f"cortical thickness ~ {corr_var_of_choice} (whole brain) is "
                f"{'' if corr_p <= params.alpha else 'not '}significant: r={r:.2f}, p={corr_p:.3g}"
            )

            if PLOT_ALL:
                sns.regplot(x=corr_var_of_choice, y="relative_global_ct", data=temp_df).set(
                    title=f"Whole brain: cortical thickness ~ {corr_var_of_choice} | r={r:.3f} | p={p:.2g}"
                )

                if SAVE_ALL:
                    path_to_plot = Path(paths.results.GM, f"corr_gct_{corr_var_of_choice}_{atl_name}.png")
                    plt.savefig(path_to_plot)
                plt.show()

        # 1. Variables of Interest
        # mean relevance per region and hemisphere + both hemispheres averaged
        print(
            "mean sum relevance per region (whole brain):\n", cs_tab.groupby("structure_name").mean()["sum_relevance"]
        )

        for hemi, hemi_table in cs_tab.groupby("hemisphere"):
            print(
                f"mean sum relevance per region in {hemi} hemisphere:\n",
                hemi_table.groupby("structure_name").mean()["sum_relevance"],
            )

        if PLOT_ALL:
            for structure_name, struc_table in cs_tab.groupby("structure_name"):
                fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, num=structure_name, figsize=(10, 5))
                for (hemi, hemi_struct_table), ax_i in zip(struc_table.groupby("hemisphere"), ax, strict=False):
                    sns.regplot(x="sum_relevance", y="gray_matter_volume_mm^3", data=hemi_struct_table, ax=ax_i)
                    ax_i.title.set_text(hemi)
                plt.show()

        # range of relevance per region and hemisphere + both hemispheres averaged
        for structure_name, struc_table in cs_tab.groupby("structure_name"):  # noqa: B007
            for hemi, hemi_table in struc_table.groupby("hemisphere"):  # noqa: B007
                hemi_table  # noqa: B018
            break

        # min/max relevance per region and hemisphere + both hemispheres averaged
        for structure_name, struc_table in cs_tab.groupby("structure_name"):
            if PLOT_ALL:
                fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, num=structure_name, figsize=(10, 5))
                for (hemi, hemi_struct_table), ax_i in zip(struc_table.groupby("hemisphere"), ax, strict=False):
                    sns.histplot(hemi_struct_table.min_relevance, ax=ax_i)
                    ax_i.title.set_text(hemi)
                plt.show()
                break

        for structure_name, struc_table in cs_tab.groupby("structure_name"):
            if PLOT_ALL:
                fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, num=structure_name, figsize=(10, 5))
                for (hemi, hemi_struct_table), ax_i in zip(struc_table.groupby("hemisphere"), ax, strict=False):
                    sns.histplot(hemi_struct_table.max_relevance, ax=ax_i)
                    ax_i.title.set_text(hemi)
                plt.show()
                break

        # sum of relevance per region and hemisphere + both hemispheres averaged

        for structure_name, struc_table in cs_tab.groupby("structure_name"):
            if PLOT_ALL:
                fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, num=structure_name, figsize=(10, 5))
                for (hemi, hemi_struct_table), ax_i in zip(struc_table.groupby("hemisphere"), ax, strict=False):
                    sns.histplot(hemi_struct_table.sum_relevance, ax=ax_i)
                    ax_i.title.set_text(hemi)
                plt.show()
                break

        for structure_name, struc_table in cs_tab.groupby("structure_name"):
            if PLOT_ALL:
                fig = plt.figure(num=structure_name, figsize=(5, 5))
                sns.histplot(struc_table.groupby("subject").sum().sum_relevance)
                plt.show()
                break

        # General overview of LIFE variables
        if PLOT_ALL:
            # predicted brain age
            plt.figure("age distribution")
            sns.histplot(cs_tab.groupby("subject").mean().brain_age, bins=82 - 18)

            # age
            plt.figure("age distribution")
            sns.histplot(cs_tab.groupby("subject").mean().age, bins=82 - 18)

            # brain age gap
            plt.figure("DBA distribution")
            sns.histplot(
                cs_tab.groupby("subject").mean().brain_age - cs_tab.groupby("subject").mean().age, bins=82 - 18
            )

            # gray-matter volume
            plt.figure("GMV distribution")
            sns.histplot(cs_tab.groupby("subject").sum()["gray_matter_volume_mm^3"], bins=82 - 18)

            # average thickness
            plt.figure("Thickness distribution")
            sns.histplot(cs_tab.groupby("subject").sum()["average_thickness_mm"], bins=82 - 18)

#  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

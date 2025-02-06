#!/usr/bin/env python3
"""
Entry point for running analysis scripts of RelevanceRelated.

Author: Simon M. Hofmann, Ole Goltermann, & Frauke Beyer | 2021-2024
"""

# %% Import
import argparse
import warnings

from relevancerelated.modeling.LRP.relevancerelated import (
    run_cs_stats,
    run_fa_stats,
    run_pvs_stats,
    run_subcortical_stats,
    run_wml_stats,
)

# %% Set paths & global vars >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Toggle to run a particular analysis
RUN_WML_ANALYSIS: bool = False  # for white matter lesions (WML), done
RUN_CS_ANALYSIS: bool = False  # for cortical surface (CS) features, done
RUN_FA_ANALYSIS: bool = False  # for fractional anisotropy (FA), done
RUN_SUB_ANALYSIS: bool = False  # for subcortical features, done
RUN_PVS_ANALYSIS: bool = False  # for perivascular spaces (PVS), done
RUN_PVS_ONLY_BORDER_ANALYSIS: bool = False  # for T1-PVS with only border-regions

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def main():
    """Run analysis scripts of RelevanceRelated."""
    # WML analysis
    if FLAGS.wml:
        run_wml_stats()

    # Cortical surface analysis
    if FLAGS.cs:
        run_cs_stats()

    # Fractional anisotropy analysis
    if FLAGS.fa:
        for mri_sequence in ("t1", "flair"):
            run_fa_stats(mri_sequence=mri_sequence)

    # Subcortical analysis
    if FLAGS.sub:
        run_subcortical_stats()

    # Perivascular spaces analysis (PVS)
    if FLAGS.pvs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # here nilearn warnings are ignored specifically
            for multi_modal in [True, False]:  # trained on multi-modal (T1-FLAIR) data or only on T1
                for mri_sequence in ["t1", "flair"]:  # take heatmaps of a sub-ensemble trained on T1 or FLAIR
                    for basal_ganglia in [True, False]:  # masked PVS with basal ganglia mask
                        for dilation in [True, False]:  # dilate PVS regions to include surrounding tissue
                            if dilation and mri_sequence == "flair":  # do this only for T1, since PVS → zero values
                                continue
                            run_pvs_stats(
                                multi_modal=multi_modal,
                                mri_sequence=mri_sequence,
                                dilate_pvs=dilation,
                                basal_ganglia=basal_ganglia,
                            )

    if FLAGS.pvs_only_border:
        from relevancerelated.configs import params

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # here we ignore nilearn warning specifically
            for multi_modal in [True, False]:  # trained on multi-modal (T1-FLAIR) data or only on T1
                for basal_ganglia in [False, True]:  # masked PVS with basal ganglia mask
                    for dilate_by in [2, 1]:
                        params.pvs.dilate_by = dilate_by
                        run_pvs_stats(
                            multi_modal=multi_modal,
                            mri_sequence="t1",
                            dilate_pvs=True,
                            basal_ganglia=basal_ganglia,
                            only_border=True,
                        )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run relevance-related analysis scripts.")

    parser.add_argument(
        "--wml",
        action=argparse.BooleanOptionalAction,
        default=RUN_WML_ANALYSIS,
        help="Run white matter hyperintensity / lesion (WML) analysis.",
    )

    parser.add_argument(
        "--cs",
        action=argparse.BooleanOptionalAction,
        default=RUN_CS_ANALYSIS,
        help="Run cortical surface (CS) analysis.",
    )

    parser.add_argument(
        "--fa",
        action=argparse.BooleanOptionalAction,
        default=RUN_FA_ANALYSIS,
        help="Run fractional anisotropy (FA) analysis.",
    )

    parser.add_argument(
        "--sub",
        action=argparse.BooleanOptionalAction,
        default=RUN_SUB_ANALYSIS,
        help="Run subcortical analysis.",
    )

    parser.add_argument(
        "--pvs",
        action=argparse.BooleanOptionalAction,
        default=RUN_PVS_ANALYSIS,
        help="Run perivascular spaces (PVS) analysis.",
    )

    parser.add_argument(
        "--pvs_only_border",
        action=argparse.BooleanOptionalAction,
        default=RUN_PVS_ONLY_BORDER_ANALYSIS,
        help="Run perivascular spaces (PVS) analysis with only border regions.",
    )

    # Parse arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Run main
    main()

# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END

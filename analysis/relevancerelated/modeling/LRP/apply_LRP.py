"""
Apply LRP on (trained) MRInet.

Author: Simon M. Hofmann | 2021
"""  # noqa: N999

# %% Import
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

warnings.simplefilter(action="ignore", category=FutureWarning)  # ignore FutureWarning especially for tf

import nibabel as nb

from relevancerelated.configs import paths
from relevancerelated.dataloader.LIFE.LIFE import get_global_max_axis, get_mni_template, prune_mri
from relevancerelated.dataloader.transformation import file_to_ref_orientation
from relevancerelated.modeling.MRInet.trained import load_trained_model
from relevancerelated.utils import cprint, load_obj

if TYPE_CHECKING:
    import numpy as np

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_sic_heatmap(
    sic: str,
    model_name: str,
    mni: bool,
    aggregated=None,  # noqa: ANN001
    analyzer_type: str = "lrp.sequential_preset_a",
    pruned_cube: bool = False,
    verbose: bool = True,
) -> np.ndarray | None:
    """Load relevance map (heatmap) of given SIC."""
    # Check which kind of model is given
    try:
        if load_trained_model(model_name).is_multilevel_ensemble():  # multi-level ensemble case
            if verbose:
                cprint(
                    string="Specify the sub-ensemble (for an aggregated heatmap) and/or basemodel "
                    "(for single model heatmap)!",
                    col="r",
                )
            return None
        # Sub-ensemble case
        if not aggregated:
            if verbose:
                cprint(
                    string=f"Since no basemodel was specified the aggregated heatmap of '{sic}' for the "
                    f"given ensemble '{model_name}' will be returned",
                    col="y",
                )
            aggregated = True
    except AttributeError:  # base-model case
        aggregated = False
        pass

    # Define file-base name
    fn = f"{analyzer_type}_relevance_maps"

    # Depending on model load different files
    if "MNI" in model_name:
        # *.nii.gz is the file in original MNI space for models trained on MNI images, i.e., not pruned,
        # whereas ".pkl.gz" is also MNI but pruned and (usually) cubified
        fn += "nii.gz" if mni else ".pkl.gz"
    else:
        # *2mni.nii.gz is the warped heatmap to the original MNI space
        fn += "2mni.nii.gz" if mni else ".pkl.gz"

    aggregated = "aggregated" if aggregated else ""
    hm_dir = Path(paths.statsmaps.LRP, model_name, aggregated, sic)

    # Load heatmap
    if mni:
        hm = nb.load(filename=Path(hm_dir, fn))

        if pruned_cube:  # prune MNI heatmap and reorient to model-training space
            # Note: this now returns a np.array instead of a nii
            hm = file_to_ref_orientation(image_file=hm)
            mni_tmp = get_mni_template(reorient=True, prune=False, mask=True)
            hm = prune_mri(
                x3d=hm.get_fdata() * mni_tmp + mni_tmp,  # mask & temp. add MNI templated
                make_cube=True,
                max_axis=get_global_max_axis(space="mni"),
            )
            hm -= get_mni_template(reorient=True, prune=True, mask=True)  # subtract MNI template again
            # adding and later subtracting MNI template is done to find proper brain-edges while pruning

    else:
        hm = load_obj(name=fn, folder=hm_dir, functimer=False)

    return hm


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END

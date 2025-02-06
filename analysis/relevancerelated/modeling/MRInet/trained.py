"""
Train and load Keras models.

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2019-2021
"""

# %% Import
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from relevancerelated.configs import paths
from relevancerelated.dataloader.LIFE.LIFE import (
    brain_regions,
    create_region_mask,
    mri_sequences,
    stamp_region,
)
from relevancerelated.utils import browse_files, cprint, split_in_n_bins

# %% Prepare data & model training  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_datasplit(model_name: str, verbose: bool = True):  # noqa: ANN201
    """Load datasplit of the model."""
    # Load data-split dict for the given model

    # Check whether basemodel of an ensemble is given
    if "Grand" in model_name:
        # Remove sub-ensemble & basemodel-part from name
        model_name = model_name.split("/")[0]
    elif "_model" in model_name:
        # Remove basemodel-part from name
        model_name = "/".join(model_name.split("/")[:-1])

    p2datasplit_model = Path(paths.keras.DATASPLIT, f"{model_name}_data_split.npy")
    if verbose:
        cprint(string=f"\nLoading data split from '{str(p2datasplit_model).split('/')[-1]}'.", col="b")
    return np.load(p2datasplit_model, allow_pickle=True).item()


def crop_model_name(model_name: str) -> str:
    """Crop model name."""
    if model_name.endswith(".h5"):
        model_name = model_name[0 : -len(".h5")]

    if model_name.endswith("_final"):
        model_name = model_name[0 : -len("_final")]

    return model_name


class TrainedEnsembleModel:
    """TrainedEnsembleModel class."""

    def __init__(self, model_name):
        """Init TrainedEnsembleModel."""
        self.path2ensemble = Path(paths.keras.MODELS, model_name).absolute()
        self.name = model_name
        self.list_of_headmodels = [hm.name for hm in Path(self.path2ensemble).iterdir() if "headmodel" in hm.name]
        self.active_model = None
        if self.is_multilevel_ensemble():
            self.list_of_submodels = sorted([
                sens.name for sens in Path(self.path2ensemble).iterdir() if sens.is_dir()
            ])
        else:
            self.list_of_submodels = sorted([
                sm.name for sm in Path(self.path2ensemble).iterdir() if "_final.h5" in sm.name
            ])

    def summary(self):
        """Get model summary."""
        # Ensemble name
        cprint(string=f"\n'{self.name}' has:", col="b")
        # Basemodels
        cprint(string=f"\n{len(self.list_of_submodels)} submodels:", fm="ul")
        print("", *self.list_of_submodels, "", sep="\n\t")
        # Headmodel(s)  # noqa: ERA001
        cprint(string=f"{len(self.list_of_headmodels)} headmodels:", fm="ul")
        print("", *self.list_of_headmodels, "", sep="\n\t")

    def is_region_ensemble(self):
        """Check if the model is a region ensemble."""
        return "region-ens" in self.name or "GrandREG" in self.name

    def is_multilevel_ensemble(self):
        """Check if the model is a multilevel ensemble."""
        return "grand" in self.name.split("/")[-1].lower()

    def get_submodel(self, submodel_name):
        """Get submodel."""
        return load_trained_model(model_name=str(Path(self.name, submodel_name)))

    def get_sics(
        self, subset: str | None = None, submodel_name: str | None = None, dropnan: bool = True, verbose: bool = False
    ) -> list[str]:
        """Get SICs."""
        if subset is not None and "train" in subset.lower() and dropnan:
            msg = "dropnan for training SICs must be implemented still!"
            raise NotImplementedError(msg)

        if submodel_name is None or not dropnan:
            sics_dict = load_datasplit(model_name=self.name, verbose=verbose)

        else:
            if self.is_multilevel_ensemble():
                submodel_name = submodel_name.split("/")[-1] if "/" in submodel_name else submodel_name
                smidx = np.where(np.array(self.list_of_submodels) == submodel_name)[0][0]
                val_sics = get_sub_ensemble_predictions(
                    model_name=self.name, subset="val", as_numpy=False, sort=True, verbose=False
                )
                val_sics = val_sics[val_sics.columns[smidx + 1]]
                val_sics = val_sics.dropna()
                val_sics = val_sics.index.to_list()
                test_sics = get_sub_ensemble_predictions(
                    model_name=self.name, subset="test", as_numpy=False, sort=True, verbose=False
                )
                test_sics = test_sics[test_sics.columns[smidx + 1]]
                test_sics = test_sics.dropna()
                test_sics = test_sics.index.to_list()

            else:
                val_sics = self.get_predictions(subset="val", submodel_name=submodel_name).sic.to_list()
                test_sics = self.get_predictions(subset="test", submodel_name=submodel_name).sic.to_list()

            sics_dict = {"validation": val_sics, "test": test_sics}

        # Return
        if subset is None:
            return sics_dict
        subset = "validation" if "val" in subset.lower() else "test" if "test" in subset.lower() else "train"
        return sics_dict[subset]

    def get_headmodel(self, head_model_type: str, multi_level: bool, cv: bool | None = None):  # noqa: ANN201
        """Get the headmodel."""
        if not self.is_multilevel_ensemble() or multi_level is False:
            cv = False
        elif cv is None:
            msg = "cv must be True OR False!"
            raise TypeError(msg)

        return load_ensemble(model_name=self.name, head_model_type=head_model_type, multi_level=multi_level, cv=cv)

    def set_active_model(self, model_name: str | None = None, verbose: bool = True):
        """Set active model."""
        if np.any([model_name in subm for subm in self.list_of_submodels]):
            # For submodel
            self.active_model = self.get_submodel(model_name)
            if verbose:
                cprint(string=f"Current active model is {self.active_model.name}", col="y")

        elif np.any([model_name in hm for hm in self.list_of_headmodels]):
            # For headmodel
            self.active_model = self.get_headmodel(
                head_model_type="nonlinear" if "_nonlin" in model_name else "linear",
                multi_level="multilevel" in model_name,
                cv="_cv_" in model_name,
            )
            if verbose:
                cprint(string=f"Current active model is {self.active_model.name}", col="y")

        else:
            cprint(string=f"No sub-/head-model with the name {model_name} found!\n > active_model = None", col="y")
            self.active_model = None

    def dt(self, _x):
        """
        Transform data.

        Data transformer (dt) for (specifically region) ensemble. dt adapts MRI data (_x) according to the active model
        in self.
        """
        if (
            self.is_region_ensemble()
            and self.active_model is not None
            and np.any([hm.rstrip("_final.h5") in self.active_model.name for hm in self.list_of_submodels])
        ):
            _xc = _x.copy()
            return stamp_region(
                dataset=_xc,
                region_mask=create_region_mask(region=get_region(self.active_model.name), reduce_overlap=True),
            )

        return _x

    def get_headmodel_data(  # noqa: ANN201
        self,
        multi_level: bool,
        subset: str = "test",
        dropnan: bool = True,
        return_sics: bool = False,
        verbose: bool = True,
    ):
        """Get headmodel data."""
        # Set subset vars correctly:
        _sset = subset if subset == "test" else "val"
        subset = subset if subset == "test" else "train"
        # since if subset = 'validation' OR = 'val': this can only be the training data for the headmodel

        # Loading data
        if not multi_level:
            if verbose:
                cprint(
                    string=f"These are the predictions on the {_sset}-set of all basemodels"
                    f"{' of each sub-ensemble' if self.is_multilevel_ensemble() else ''}!",
                    col="b",
                )

            # Train (i.e., predictions on the validation set!) OR test-set
            p2df = Path(paths.keras.MODELS, f"{self.name}/{subset.lower()}_data_ensemble_head.sav")
            _x, _y = joblib.load(filename=p2df)  # xdata, ydata
            # if return_sics:
            _sset = _sset if _sset == "test" else "validation"
            sics_subset = np.array(load_datasplit(model_name=self.name, verbose=verbose)[_sset])

        else:  # multi_level:
            # Check arguments
            if not self.is_multilevel_ensemble():
                msg = "A non-multi-level ensemble has no multi-level headmodel data! Return None!"
                raise ValueError(msg)
            if verbose:
                cprint(string=f"These are the predictions on the {_sset}-set of all sub-ensembles!", col="b")
            data = get_sub_ensemble_predictions(
                model_name=self.name, subset=_sset, as_numpy=False, sort=True, verbose=verbose
            )
            sics_subset = data.index.to_numpy()
            _y = data.pop(f"{_sset}_y").to_numpy().astype("float32")
            _x = data.to_numpy().astype("float32")

        # Drop NaN values
        if dropnan:
            (_x, _y), nan_idx = remove_nan_ensemble_data(_x, _y, return_nan_index=True)
            sics_subset = np.delete(arr=sics_subset, obj=[] if nan_idx is None else nan_idx)
            if not len(_x) == len(_y) == len(sics_subset):
                msg = "Length must be equal, revisit implementation!"
                raise ValueError(msg)

        # Return
        if return_sics:
            return _x, _y, sics_subset
        return _x, _y

    def get_headmodel_predictions(self, return_single_performs: bool = False, verbose: bool = True):  # noqa: ANN201
        """
        Get concatenated predictions of linear headmodels trained via CV.

        Works only for multi-level ensembles.
        Note that the order of the returned data is different to self.get_headmodel_data(...) due to the
        randomization in the CV split.
        The corresponding mixing indices can be found in the dict returned via
        self.get_headmodel(..., cv=True)
        """
        if not self.is_multilevel_ensemble():
            cprint("This works only for multi-level ensembles", "r")
            return None

        cprint("Get predictions of linear headmodel which was trained via cross-validation...", "b")
        # Note that the order differs

        xdata, ydata, sics = self.get_headmodel_data(
            multi_level=True, subset="test", dropnan=True, return_sics=True, verbose=False
        )
        cv_headmodels = self.get_headmodel(head_model_type="linear", multi_level=True, cv=True)["head_models"]

        split_indices = self.get_headmodel_cv_sort_indices(concat=False)
        sort_indices = np.concatenate(split_indices)  # == self.get_headmodel_cv_sort_indices(concat=True)
        # necessary since split_in_n_bins() changes order when remainders are present

        # # Collect predictions
        # Concatenate all CV test-sets (of the best split-set), and the corresponding model predictions
        all_preds = None
        all_testy = None  # only for testing

        maes = []  # Collect performances per split
        r2s = []
        for sp in range(len(split_indices)):
            test_indices = split_indices[sp].copy()
            _testx = xdata[test_indices].copy()
            _testy = ydata[test_indices].copy()
            _preds = cv_headmodels[sp].predict(_testx).copy()

            # Collect predictions over different splits (note different orders)
            all_preds = _preds if all_preds is None else np.concatenate([all_preds, _preds])
            all_testy = _testy if all_testy is None else np.concatenate([all_testy, _testy])
            # Note: Adapt order of data: xdata[sort_indices] ~ ydata[sort_indices] ~ all_preds

            maes.append(np.mean(np.abs(_preds - _testy)))  # MAE per split
            r2s.append(cv_headmodels[sp].score(_testx, _testy))  # R2

        if verbose:
            cprint(string=f"MAE = {np.mean(maes):.2f} (r2 = {np.mean(r2s):.2f})", col="b")

        # Return as dataframe
        pred_df = pd.DataFrame({"predictions": all_preds, "sics_sorted": sics[sort_indices], "target": all_testy})

        if return_single_performs:
            return pred_df, maes, r2s
        return pred_df

    def get_headmodel_cv_sort_indices(self, concat: bool = True):  # noqa: ANN201
        """Get headmodel CV sort indices."""
        if not self.is_multilevel_ensemble():
            cprint(string="This works only for multi-level ensembles", col="r")
            return None

        data_indices = self.get_headmodel(head_model_type="linear", multi_level=True, cv=True)["data_indices"]

        split_indices = split_in_n_bins(a=data_indices, attribute_remainder=True)
        if concat:
            return np.concatenate(split_indices)
        return split_indices

    def get_predictions(  # noqa: ANN201
        self, subset: str = "test", submodel_name: str | None = None, verbose: bool = True
    ):
        """Get predictions."""
        if self.active_model is None and submodel_name is None:
            cprint(string="Activate a submodel first OR provide the name of the sub/head-model!", col="r")
            return None

        subset = subset.lower()
        subset = "val" if "val" in subset else "test" if "test" in subset else None
        if subset is None:
            msg = "Subset must be 'val' OR 'test'!"
            raise ValueError(msg)

        if isinstance(self.active_model, TrainedEnsembleModel) or (
            isinstance(submodel_name, str) and "_ens" in submodel_name
        ):
            # Return predictions of the linear headmodel of the given sub-ensemble
            data = get_sub_ensemble_predictions(model_name=self.name, subset=subset, as_numpy=False)
            # Since the order of predictions (columns) is not necessarily the order of self.list_of_submodels,
            # we must extract the right prediction column

            # Define MRI (and Region) to drop from the table:
            if submodel_name is None:
                submodel_name = self.active_model.name.split("/")[-1]

            drop_seq = [seq for seq in mri_sequences if seq.upper() not in submodel_name]
            drop_reg = [
                reg.upper()[0:3]
                for reg in np.array(sorted(brain_regions))[[0, 2, 1]]
                if (reg.upper()[0:3] not in submodel_name)
            ]

            # Throw out sequences (and regions) which do not match the given sub-ensemble
            keep_cols = data.columns.to_list()
            for dropper in [drop_seq, drop_reg]:
                for d in dropper:
                    while True:
                        for i, c in enumerate(keep_cols):
                            if d in c:
                                keep_cols.pop(i)
                                break
                        else:
                            break

            if len(keep_cols) > 2:  # noqa: PLR2004
                # Remove "pred" column if required sequence(-region pair) is found in another column
                keep_cols.pop(np.where(np.array(keep_cols) == "pred")[0][0])

            # Sort analogous to return for basemodel predictions
            data = data[sorted(keep_cols)]  # 'pred...' before 'test/val_y'
            data = data.reset_index()  # make SIC-index to column

            if verbose:
                cprint(
                    f"Return predictions of linear head-model of sub-ensemble:\n{self.name}/{submodel_name}",
                    col="b",
                )
                print("To get predictions of all sub-ensembles, use:\n\tself.get_headmodel_data() OR ...")
                print(
                    "To get predictions of a requested base model, use:\n\t"
                    "self.active_model.get_predictions(submodel_name=BASE_MODEL_NAME)\n"
                )

            return data

        if self.active_model is None:
            pred_file = crop_model_name(submodel_name)
        else:
            pred_file = self.active_model.name.split("/")[-1]
        pred_file += f"_{subset}_predictions.csv"

        for file in Path(self.path2ensemble).iterdir():
            if file.name == pred_file:
                file = pd.read_csv(file)  # noqa: PLW2901
                break
        else:
            file = None
            print(f"No {subset}-prediction file was found for '{self.active_model.name}'.")
            # compute the file ...

        return file

    def get_mri_sequence(self):
        """Get MRI sequence."""
        if self.active_model is None and "sequence-ens" in self.name:
            cprint(string="Activate a submodel first, since you operate with a sequence-ensemble!", col="r")
            return None
        if "sequence-ens" in self.name:
            return self.active_model.name.split("_")[-2]
        for seq in mri_sequences:
            if seq.upper() in self.name:
                return seq
        print("No indication for other than T1 MRI sequence found! Return 't1'!")
        return "t1"

    def get_region(self):
        """Get region."""
        if self.is_region_ensemble() and self.is_multilevel_ensemble():
            if self.active_model is not None:
                return next(
                    r for r in np.array(sorted(brain_regions))[[0, 2, 1]] if (r[:3].upper() in self.active_model.name)
                )
            cprint(string="Activate a submodel first to determine the region it is trained on!", col="r")
            return None
        if self.is_region_ensemble() and not self.is_multilevel_ensemble():
            if "GrandREG" in self.name:  # in case it is a sub-ensemble of Grand-Ensemble
                return next(r for r in np.array(sorted(brain_regions))[[0, 2, 1]] if (r[:3].upper() in self.name))
            if self.active_model is not None:
                return next(r for r in np.array(sorted(brain_regions))[[0, 2, 1]] if (r in self.active_model.name))
            cprint("Activate a submodel first to determine the region it is trained on!", "r")
            return None
        cprint("Model was trained on whole brain data, hence there is no specific region!", "r")
        return None

    def get_n_params(self, return_n: bool = False, verbose: bool = True):  # noqa: ANN201
        """Get n params."""
        if self.is_multilevel_ensemble():
            n_subens = len(self.list_of_submodels)  # n sub-ensembles
            sub_ens = self.get_submodel(self.list_of_submodels[0])
            n_bm = len(sub_ens.list_of_submodels)  # n basemodels
            bm = sub_ens.get_submodel(sub_ens.list_of_submodels[0])  # load one basemodel to count params
            n_params_bm = bm.count_params()  # n params in a basemodel
            prt_line = f"{n_subens} sub-ensembles with each "  # extra text for printing

        else:
            n_subens = 1  # 1 for later multiplication, actually zero
            n_bm = len(self.list_of_submodels)
            bm = self.get_submodel(self.list_of_submodels[0])
            n_params_bm = bm.count_params()  # n params per basemodel
            prt_line = ""

        # Calculate all params
        all_n_params = n_params_bm * n_bm * n_subens

        if verbose:
            cprint(
                string=f"'{self.name}' has {prt_line}{n_bm} basemodels. "
                f"Each basemodel has {n_params_bm} parameters. \n "
                f"Hence, the whole ensemble has {all_n_params} parameters (excl. headmodel(s)).",
                col="b",
            )

        if return_n:
            return all_n_params
        return None


def load_trained_model(model_name: str | None = None) -> TrainedEnsembleModel | tf.keras.models.Sequential:
    """Load trained model."""
    if model_name:
        if Path(paths.keras.MODELS, model_name).is_dir():
            # Load & return ensemble model
            return TrainedEnsembleModel(model_name=model_name)

        # Load single model
        if ".h5" not in str(model_name) and "_final" not in str(model_name):
            model_name += "_final.h5"
        return tf.keras.models.load_model(Path(paths.keras.MODELS, model_name))

    cprint(
        string="Note: If you browse for an ensemble model, just choose a random submodel from the "
        "respective ensemble model folder",
        col="y",
    )

    path2model = browse_files(paths.keras.MODELS, "H5")

    if "_ens" in path2model or "region-ens" in path2model:
        _model_name = path2model.split("/")[-2]

    else:
        _model_name = path2model.split("/")[-1]

    return load_trained_model(model_name=_model_name)


# %% Model ensembles  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_sub_ensemble_predictions(  # noqa: ANN201
    model_name: str, subset: str, as_numpy: bool, sort: bool = True, verbose: bool = False
):
    """Compute the predictions of all sub-ensembles of grand-ensemble in given subset."""
    subset = subset.lower()
    if subset not in {"val", "validation", "test"}:
        msg = "sub-ensemble predictions are only available on the validation or test set!"
        raise ValueError(msg)

    if subset == "validation":
        subset = "val"

    # Check if model_name is name of grand-ensemble
    if "grand" not in model_name.split("/")[-1].lower():
        msg = "This functions works only for grand-ensemble models!"
        raise ValueError(msg)

    p2pred = Path(paths.keras.MODELS, model_name)
    p2sav = p2pred / f"all_subens_preds_on_{subset}set.sav"
    max_col = f"{subset}_y"

    if p2sav.is_file():
        pred_tabs = joblib.load(filename=p2sav)

    else:
        # Find sub-ensembles
        pred_tabs = None  # init
        i = -1  # indexer
        _sset = subset if subset == "test" else "train"

        for p2sub_ens in Path(p2pred).iterdir():
            if p2sub_ens.is_dir():
                i += 1

                # Load sub-ensemble
                sub_ens = TrainedEnsembleModel(model_name=str(p2sub_ens).lstrip(paths.keras.MODELS))

                # Use only linear headmodel
                sub_ens_hm = sub_ens.get_headmodel(head_model_type="linear", multi_level=False, cv=False)

                # Load train or test data of sub-ensemble
                sub_hm_data = sub_ens.get_headmodel_data(multi_level=False, subset=_sset, dropnan=False, verbose=False)
                # There are no NaNs in this data

                # Get also the corresponding SICs, since data-length between sub-ensembles can differ
                sics_data = pd.read_csv(
                    Path(sub_ens.path2ensemble, f"0_model_{subset}_predictions.csv"),
                    # for all base models the same
                    index_col="sic",
                )
                sub_ens_pred_tab = sics_data.drop(columns=f"{subset}_pred", inplace=False)

                if not np.all(sub_hm_data[1] == sics_data[f"{subset}_y"].to_numpy()):
                    msg = "must be the same ..."  # [TESTED]
                    raise ValueError(msg)

                # Get sub-ensemble predictions on subset
                sub_ens_pred = sub_ens_hm.predict(sub_hm_data[0])
                sub_ens_pred_tab["pred"] = sub_ens_pred
                # This will become training data for top-headmodel

                if verbose:
                    # TODO: if following will be kept, do also for classification  # noqa: FIX002
                    cprint(
                        string=f"{subset.title()}-performance of sub-ensemble {sub_ens.name.split('/')[-1]}: "
                        f"{np.mean(np.abs(sub_hm_data[1] - sub_ens_pred)):.3f}",
                        col="y",
                    )  # TEST

                # Save predictions of all sub-ensembles in common df
                if pred_tabs is None:
                    pred_tabs = sub_ens_pred_tab
                else:
                    sfx = f"_{sub_ens.get_mri_sequence()}"
                    sfx += sub_ens.get_region().upper()[:3] if sub_ens.is_region_ensemble() else ""
                    pred_tabs = pred_tabs.join(other=sub_ens_pred_tab, on="sic", rsuffix=sfx)

        # Find most complete y-col
        for col in pred_tabs.columns:
            if "_y" in col:
                max_col = col if pred_tabs[col].count() > pred_tabs[max_col].count() else max_col

                if col != max_col:  # Remove y-data from df which is less complete
                    pred_tabs = pred_tabs.drop(columns=col)

    if sort:
        y_idx = [i for i, elem in enumerate(pred_tabs.columns) if max_col == elem]  # == [0], index of y
        preds_idx = [i for i, elem in enumerate(pred_tabs.columns) if "pred" in elem]  # indices of preds

        colorder = []  # init, for proper order
        if "REG" in model_name and len(preds_idx) == 3 * 3:
            # Regional ensemble
            for _reg in np.array(sorted(brain_regions.keys()))[[0, 2, 1]]:  # [CER, SUB, COR]
                reg = _reg[0:3].upper()
                for seq in mri_sequences:
                    seq_idx = [i for i, elem in enumerate(pred_tabs.columns) if (seq in elem and reg in elem)]
                    if len(seq_idx) == 0:  # without sequence-suffix in column-name
                        # Get index that is not the target col and has no info about region and sequence
                        seq_idx = [
                            i
                            for i, elem in enumerate(pred_tabs.columns)
                            if (
                                all(s not in elem for s in mri_sequences)
                                and all(b[:3].upper() not in elem for b in brain_regions)
                                and i not in y_idx
                            )
                        ]
                    colorder.append(seq_idx[0])

        elif "REG" not in model_name and len(preds_idx) == 3:  # noqa: PLR2004
            # Whole-brain grand-ensemble

            for seq in mri_sequences:
                seq_idx = [i for i, elem in enumerate(pred_tabs.columns) if seq in elem]
                if len(seq_idx) == 0:  # without sequence-suffix in column-name
                    # Get index that is not the target col and has no info about the mri sequence
                    seq_idx = [
                        i
                        for i, elem in enumerate(pred_tabs.columns)
                        if (all(s not in elem for s in mri_sequences) and i not in y_idx)
                    ]

                colorder.append(seq_idx[0])

        else:
            msg = f"Prediction table of '{model_name}' has unknown structure!"
            raise ValueError(msg)

        # Sort columns according to
        # * MRI sequences (1.T1, 2.FLAIR, 3.SWI), and, if applicable, according to
        # * brain region (1.CER, 2.SUB, 3.COR)
        pred_tabs = pred_tabs[pred_tabs.columns[y_idx + colorder]]

    if verbose:
        cprint(f"Column-order of sub-ensemble predictions: {[p for p in pred_tabs.columns if 'pred' in p]}", col="y")

    # Pull ydata
    if as_numpy:
        ydata_ = pred_tabs.pop(max_col).to_numpy().astype(np.float32)  # (N, )
        x_ensemble_ = pred_tabs.to_numpy().astype(np.float32)  # (N, M)

        return x_ensemble_, ydata_

    return pred_tabs


def remove_nan_ensemble_data(  # noqa: ANN201
    _xdata: np.ndarray, _ydata: np.ndarray, return_nan_index: bool = False
):
    """Remove NaN's in ensemble data."""
    nan_idx = None
    if np.isnan(_xdata).any():
        nan_idx = np.where(np.isnan(_xdata).any(axis=1))[0]
        # Delete rows
        _ydata = _ydata[~np.isnan(_xdata).any(axis=1)]  # start with y-values first
        _xdata = _xdata[~np.isnan(_xdata).any(axis=1)]

    if return_nan_index:
        return (_xdata, _ydata), nan_idx
    return _xdata, _ydata


def load_ensemble(  # noqa: ANN201
    model_name: str, head_model_type: str = "linear", multi_level: bool = False, cv: bool = False
):
    """Load ensemble."""
    if cv and (not multi_level or head_model_type != "linear"):
        cprint(string="All cross-validated models are always linear (multi-level) ensemble headmodels", col="y")
        # Set args correctly:
        multi_level = True
        head_model_type = "linear"

    if head_model_type.lower() == "linear":
        ensemble_model = joblib.load(
            filename=Path(
                paths.keras.MODELS,
                model_name,
                f"ensemble_{'multilevel_' if multi_level else ''}{'cv_' if cv else ''}headmodel_lin.sav",
            )
        )

    elif head_model_type.lower() == "nonlinear":
        ensemble_model = load_trained_model(
            model_name=f"{model_name}/ensemble_{'multilevel_' if multi_level else ''}headmodel_nonlin.h5"
        )

    else:
        msg = "'head_model_type' unknown. Must be 'linear' OR 'nonlinear'."
        raise ValueError(msg)

    return ensemble_model


# %% Get data specific to a (trained) model ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_region(_model_name):
    """Get region."""
    for reg in np.array(sorted(brain_regions))[[0, 2, 1]]:  # CER, SUB, COR
        if f"_{reg.upper()[0:3]}_" in _model_name or f"_{reg}_" in _model_name:
            region = reg
            break
    else:
        region = None

    return region


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END

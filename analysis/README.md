# RelevanceRelated – analysis

    Period: September 2021 - December 2025
    Where: MPI CBS, Leipzig, Germany

| Authors      | Simon M. Hofmann            | Ole Goltermann           | Frauke Beyer          |
|:-------------|:----------------------------|:-------------------------|:----------------------|
| **Contact**  | simon.hofmann[ät]cbs.mpg.de | o.goltermann[ät]uke.de   | fbeyer[ät]cbs.mpg.de  |

![Last update](https://img.shields.io/badge/where-MPI_CBS-green)
![Last update](https://img.shields.io/badge/last_update-Feb_06,_2025-blue)

---

Scripts for processing and analysis are combined in a Python package `relevancerelated`
in [`./relevancerelated/`](./relevancerelated).
This includes the processing and analysis of MRIs, brain features
and relevance maps generated from the brain-age prediction models.
To install the package, refer to the [README](../README.md#install-the-research-project-package-relevancerelated)
in the root folder of the repository.[^1]

The following modules of the package are briefly described:

## dataloader

[`relevancerelated.dataloader`](./relevancerelated/dataloader/)
contains code to load and preprocess various MRI datasets.

## modeling

### Perivascular spaces (PVS) segmentation

PVS is extracted from T1 & FLAIR images.

A bash script to run the PVS segmentation can be found in [`./scripts/`](./scripts/).
This script makes use of the code in
[`relevancerelated.modeling.pvs_extraction`](./relevancerelated/modeling/pvs_extraction.py).
PVS segmentation is based on [SHIVA_PVS](https://github.com/pboutinaud/SHIVA_PVS/blob/main/predict_one_file.py).


### MRInet

[`relevancerelated.modeling.MRInet.trained`](./relevancerelated/modeling/MRInet/trained.py) contains loader functions
for the 3D-convolutional neural networks (CNN) models that were trained to predict brain-age from MRI images,
and were then subject to the XAI analysis (using [LRP](README.md#lrp)) generating relevance maps.
For details see our previous study [Hofmann et al. (*NeuroImage*, 2022)](https://doi.org/10.1016/j.neuroimage.2022.119504).

### LRP

Submodule [`relevancerelated.modeling.LRP`](./relevancerelated/modeling/LRP/) contains scripts for the application
of the XAI-method Layer-wise relevance propagation (
[LRP; Bach et al., 2015]((https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)))
on MRI-based prediction models.
As a post-hoc XAI-method, LRP highlights information in the input space
being relevant for the given model prediction. <br>
Here, the LRP is applied to the predictions of the [MRInet](README.md#mrinet)
extracting voxels in the MRI image that were relevant for brain-age estimations. <br>
The code is built around the [iNNvestigate](https://github.com/albermax/innvestigate) toolbox.


In this project, we related relevance maps with other brain features (e.g., PVS, cortical thickness, etc.),
which is mainly done in the script [`relevancerelated.py`](./relevancerelated/modeling/LRP/relevancerelated.py)
in the `LRP` submodule and subsequently applied [R scripts](Rscripts).

The pipeline of `relevancerelated.py` can be run with the following line (use the `--help` flag for an overview):

```shell
python relevancerelated.run --help
```

### Statistical analysis

Statistical analyses are primarily performed in `R` (see the folder [./Rscripts/](Rscripts)),
and some in Python (see [relevancerelated.statistics](./relevancerelated/statistics/)).

## Configuration

Most of the project configuration is collected in the [`./configs/config.toml`](./configs/config.toml) file [^2].
The file is read out with the script [`configs.py`](./relevancerelated/configs.py).

> The general project structure follows the [**scilaunch**](https://shescher.github.io/scilaunch/) template.

## Applying the pipeline on your own data

If you are interested in training deep learning models and applying relevance mapping (XAI) on your own data,
we refer to the [`xai4mri`](https://shescher.github.io/xai4mri/) toolbox,
which is a generalized form of the code presented here.

## COPYRIGHT/LICENSE

See the [LICENSE](../LICENSE) file for details.

---

[^1]: *Note, some code is anonymized to avoid data leakage, and is primarily intended for demonstration purposes.
For data access, get in touch with us.*

[^2]: *Note, some paths are anonymized here.*

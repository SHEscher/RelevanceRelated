# RelevanceRelated

![Last update](https://img.shields.io/badge/where-MPI_CBS-green)
![Last update](https://img.shields.io/badge/version-v.2.0.0-blue)

## Briefly

This repository contains the code for our study ([Hofmann & Goltermann et al., 2025](README.md#citation)) that utilized participant-level XAI-based relevance maps
(using *Layer-wise Relevance Propagation*, [LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140))
derived from two ensembles of 3D-convolutional neural networks
trained on T1-weighted and fluid attenuated inversion recovery images (FLAIR) of 2016 participants, aged 18-82 years
(as reported in [Hofmann et al., 2022)](https://doi.org/10.1016/j.neuroimage.2022.119504).
These relevance maps were associated with various human-interpretable structural brain features,
including regional cortical and subcortical gray matter volume and thickness, perivascular spaces,
and water diffusion-based fractional anisotropy of primary white matter tracts.
The approach aimed to bridge voxel-based contributions to brain-age estimates with biologically meaningful
structural features of the aging brain.

## Project structure

Refer to READMEs in the subfolders for more information.
For instance, the project code can be found in `analysis/`
(for more information check the corresponding [README](./analysis/README.md)).

## Installation of the research code

The research code can be easily installed, together with its dependencies as a `Python` package,
following the commands below. Additional `R` scripts can be found and executed in `analysis/Rscripts/`.
For `R`, it is recommended to open the project (e.g., via `RStudio`) on the root level.

### Create an environment

Use `Python3.10` and install the necessary packages preferably in a `conda` environment:

```shell
# create a new environment specific to this project
conda create -n relrel_3.10 python=3.10
```

Activate the environment:

```
conda activate relrel_3.10
```

### Install the research project package `relevancerelated`

Before running the line below in your terminal, make sure that you are in the root folder `RelevanceRelated/`,
where this README file lies.
Also, make sure that the **root** folder has the *CamelCase* naming.
Some operating systems do not keep this after cloning repositories from a remote server.

Install the study code as `Python` package:

```shell
# while the conda environment is active & you are in the root folder:
pip install -e .
```

Note, there are optional dependencies for those who want to develop this project further.
Check out the [`setup.cfg`](./setup.cfg) file to see which dependencies will be installed additionally
(add the `[develop]` flag to the `pip` installation line above).

Also, note that some of the code had to be anonymized due to data protection regulations.
Therefore, some parts of the code might not be executable without the original data.
Contact the authors for information on how to get access to the data.

### Applying the pipeline on your own data

If you are interested in training deep learning models and applying relevance mapping (XAI) on your own data,
we refer to the [`xai4mri`](https://shescher.github.io/xai4mri/) toolbox,
which is a generalized form of the code presented here.

## Versions

### version >= `v.2.0.0`
* fine-tuned GMV, CS, PVS and FA analysis, and more
* related publication [Hofmann & Goltermann et al. (Imaging Neuroscience, 2025)](#citation)

### version < `v.2.0.0`

* GMV, CS and FA analysis were presented at OHBM 2023
* WML analysis was published in [Hofmann et al. (2022)](https://doi.org/10.1016/j.neuroimage.2022.119504)

## Citation

If you use code or data from this repository, please cite the following paper:

[Hofmann, S.M., Goltermann, O., Scherf, N., Müller, K.R., Löffler, M., Villringer, A., Gaebler, M., Witte, A.V., Beyer, F. The utility of explainable AI for MRI analysis: Relating model predictions to neuroimaging features of the aging brain. *Imaging Neuroscience*. 2025.](https://doi.org/10.1162/imag_a_00497)

```bibtex
@article{hofmanngoltermannUtilityExplainableAI2025,
    author = {Hofmann, Simon M. and Goltermann, Ole and Scherf, Nico and Müller, Klaus-Robert and Löffler, Markus and Villringer, Arno and Gaebler, Michael and Witte, A. Veronica and Beyer, Frauke},
    title = {The utility of explainable AI for MRI analysis: Relating model predictions to neuroimaging features of the aging brain},
    journal = {Imaging Neuroscience},
    volume = {3},
    pages = {imag_a_00497},
    year = {2025},
    month = {02},
    issn = {2837-6056},
    doi = {10.1162/imag_a_00497},
    url = {https://doi.org/10.1162/imag\_a\_00497},
    eprint = {https://direct.mit.edu/imag/article-pdf/doi/10.1162/imag\_a\_00497/2503311/imag\_a\_00497.pdf},
}
```

---

The authors of this repository are:

| Authors      | Simon M. Hofmann            | Ole Goltermann           | Frauke Beyer          |
|:-------------|:----------------------------|:-------------------------|:----------------------|
| **Contact**  | simon.hofmann[ät]cbs.mpg.de | o.goltermann[ät]uke.de   | fbeyer[ät]cbs.mpg.de  |

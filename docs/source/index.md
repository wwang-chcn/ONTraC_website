# Welcome to the ONTraC (Ordered Niche Trajectory Construction) Website

<span>
  <a href="https://pypi.org/project/ONTraC/">
    <img src="https://img.shields.io/pypi/v/ONTraC.svg" alt="PyPI version" style="display:inline-block;">
  </a>
  <a href="https://pypi.org/project/ONTraC/">
    <img src="https://img.shields.io/pypi/pyversions/ONTraC.svg" alt="Python versions" style="display:inline-block;">
  </a>
  <a href="https://pepy.tech/project/ONTraC">
    <img src="https://static.pepy.tech/badge/ONTraC" alt="Downloads" style="display:inline-block;">
  </a>
  <a href="https://github.com/gyuanlab/ONTraC">
    <img src="https://badgen.net/github/stars/gyuanlab/ONTraC" alt="GitHub Stars" style="display:inline-block;">
  </a>
  <a href="https://github.com/gyuanlab/ONTraC/issues">
    <img src="https://img.shields.io/github/issues/gyuanlab/ONTraC.svg" alt="GitHub Issues" style="display:inline-block;">
  </a>
  <a href="https://github.com/gyuanlab/ONTraC/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/gyuanlab/ONTraC.svg" alt="GitHub License" style="display:inline-block;">
  </a>
</span>

## Overview

ONTraC (Ordered Niche Trajectory Construction) is a niche-centered machine learning method for constructing spatially continuous trajectories.

ONTraC differs from existing tools in that it treats a niche, rather than an individual cell, as the basic unit for spatial trajectory analysis. A niche is defined as a multicellular, spatially localized region where different cell types may coexist and interact with each other.

ONTraC integrates niche features (cell-type composition) and spatial information using a graph neural network framework. Its output, the niche trajectory, represents a one-dimensional continuum of the tissue microenvironment. By disentangling cell-level and niche-level properties, niche trajectory analysis offers a coherent framework for studying coordinated cellular responses to continuous tissue microenvironment variations.

ONTraC can assign every cell from all samples an NT (niche trajectory) score to describe its position along the niche trajectory, which provides a uniform spatial trajectory across the whole dataset.

```{image} ../source/images/other/logo_with_text_long.png
:alt: ONTraC logo
```

```{image} ../source/images/other/ONTraC_structure.png
:alt: ONTraC structure
```

## Quick start

### Input File

This file contains all input information with five columns: `Cell_ID`, `Sample`, `Cell_Type`, `x`, and `y`.

| Cell_ID         | Sample   | Cell_Type | x       | y     |
| --------------- | -------- | --------- | ------- | ----- |
| E12_E1S3_100034 | E12_E1S3 | Fibro     | 15940   | 18584 |
| E12_E1S3_100035 | E12_E1S3 | Fibro     | 15942   | 18623 |
| ...             | ...      | ...       | ...     | ...   |
| E16_E2S7_326412 | E16_E2S7 | Fibro     | 32990.5 | 14475 |

For detailed information about input and output files, please see the [IO Files Explanation](./tutorials/IO_files.md).

### Installation

```bash
pip install "ONTraC[all]"
```

- See the [Installation and Setup](./installation.md) for more details.

### Run ONTraC

```bash
ONTraC \
    --meta-input simulated_dataset.csv \
    --NN-dir simulation_NN \
    --GNN-dir simulation_GNN \
    --NT-dir simulation_NT \
    --hidden-feats 4 \
    -k 6 \
    --modularity-loss-weight 0.3 \
    --purity-loss-weight 300 \
    --regularization-loss-weight 0.1 \
    --beta 0.03 2>&1 | tee simulation.log
```

- See the [Command Line Interface Tutorial](./tutorials/command_line.md) for using ONTraC from the command line.
- For Jupyter-based usage, refer to the [Interactive Tutorial](./tutorials/interactive.ipynb).

### Visualization

```bash
ONTraC_analysis \
    -o analysis_output/simulation \
    -l simulation.log \
    --NN-dir simulation_NN \
    -GNN-dir simulation_GNN \
    --NT-dir simulation_NT \
    -r
```

- See the [Visualization Tutorial](./tutorials/visualization.ipynb) for more details.

```{note}
This project is actively maintained and under active development.
```

## Citation

**Wang, W.\*, Zheng, S.\*, Shin, C. S., Chávez-Fuentes J. C. & [Yuan, G. C.](https://labs.icahn.mssm.edu/yuanlab/)$**. [ONTraC characterizes spatially continuous variations of tissue microenvironment through niche trajectory analysis.](https://doi.org/10.1186/s13059-025-03588-5) *Genome Biol*, 2025.

## Other Resources

[Reproducible code for our paper](https://github.com/gyuanlab/ONTraC_paper)

[Datasets and outputs used in our paper](https://doi.org/10.5281/zenodo.11186619)

```{toctree}
:caption: Contents
:hidden: true
:maxdepth: 2
name: maintoc
numbered:

Home <self>
installation
tutorials/index
examples/index
how_it_works/index
contributors
citation
```

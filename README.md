# Reproducible Figures for *Pure Steady-State CEST*

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.mri.2025.110506-blue)](https://doi.org/10.1016/j.mri.2025.110506)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)

This repository contains Python scripts for reproducing selected figures and example reconstruction results from the publication:

> **Pure steady-state CEST**  
> Johannes Hammacher, Christoph Kolbitsch, and Patrick Schünke  
> *Magnetic Resonance Imaging*, 2025  
> [https://doi.org/10.1016/j.mri.2025.110506](https://doi.org/10.1016/j.mri.2025.110506)

## Overview

The repository provides examples for:

- generating example Pulseq sequences used in the study.
- reproducing selected figures from the publication;
- reconstructing raw data using a representative reconstruction pipeline;

The scripts are designed to be run independently wherever possible. The corresponding figures are automatically saved to disk, typically as `.png` or `.pdf` files.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>

pip install -r requirements.txt
```

A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Reproduce publication figures

To reproduce a figure, run the corresponding Python script:

```bash
python recreate_figure_*.py
```

Each script generates the relevant output and saves it to disk. Please refer to the individual scripts for details about input data, parameters, and output locations.

## Reconstruct raw data

An example reconstruction pipeline is provided in:

```bash
python reconstruction_example.py
```

This script demonstrates the basic steps required to reconstruct data acquired with the steady-state CEST approach.

## Generate Pulseq sequences

Example Pulseq sequences used in the publication can be generated with the corresponding `write_CEST*` scripts:

```bash
python write_CEST<sequence_name>.py
```

The generated sequence files can then be inspected and the publicaton figures recreated.

## Repository structure

A typical workflow is:

```text
.
├── requirements.txt
├── write_CEST*.py
├── reconstruction_example.py
├── recreate_figure_*.py
└── README.md
```

## Citation

If you use this code or reproduce results from this repository, please cite:

```text
Hammacher J, Kolbitsch C, Schünke P. Pure steady-state CEST.
Magnetic Resonance Imaging. 2025.
doi:10.1016/j.mri.2025.110506.
```

BibTeX:

```bibtex
@article{Hammacher2025Pure,
  author  = {Hammacher, Johannes and Kolbitsch, Christoph and Sch{\"u}nke, Patrick},
  title   = {Pure steady-state CEST},
  journal = {Magnetic Resonance Imaging},
  year    = {2025},
  doi     = {10.1016/j.mri.2025.110506}
}
```

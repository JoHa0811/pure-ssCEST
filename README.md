# Reproducible Figures for Pure steady-state CEST

This repository contains simple Python scripts used to recreate selected figures from the publication:

**Pure Steady-State CEST**  
Johannes Hammacher, Christoph Kolbitsch, Patrick Schünke
[DOI or arXiv link]

## Overview

Each script in this repository is self-contained and can be run independently. When executed, the corresponding figure from the publication will be generated automatically and saved to disk (typically as a `.png` or `.pdf` file).

## Requirements

To install the dependencies, you can run:
```bash
pip install -r requirements.txt
````

## Recreate publication figures
To recreate the respective figure from the publication, simply run the corresponding script.

## Reconstruct raw data
For an example of the reconstruction pipeline, please see reconstrucion_example.py

## Generate Pulseq sequences
To generate example sequences used in the publication, simply run the respective write_CEST* script.

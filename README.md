# Differentiable Tau Propagation (Minimal Release)

This repository provides a **minimal, self-contained implementation** of the core computational framework introduced in our manuscript/preprint:

> *Spatially resolved mapping of tau amplification rates via differentiable simulation of prion-like propagation*

---

## Overview

This repository focuses on methodological contributions of the paper:

* Differentiable reaction–diffusion simulation of prion-like propagation
* MRI-informed model parametrization
* GPU-accelerated implementation using JAX
* End-to-end differentiability for gradient-based optimization

Due to the use of restricted neuroimaging datasets (e.g., ADNI), reproducing the complete analysis pipeline requires controlled data access.
Instead, this release is intended to facilitate evaluation and reuse of the core methodological components of the work.

---

## What is included

* `simndd/`
  Core library implementing the differentiable reaction–diffusion simulator

* `forward_simulation.ipynb`
  A minimal demonstration of:

  * Processing of neuroimages for model parametrization
  * Forward simulation of tau propagation
  * Visualization of spatiotemporal dynamics

---

## What is not included

The following components depend on restricted datasets (e.g., ADNI) and are **not included in this release**:

* Tau PET data processing pipeline
* Subject-level parameter inference scripts
* Full training / optimization pipeline used in the paper

Additional components may be released in future versions, subject to data-sharing agreements and institutional policies.

---

## Data availability

The original study uses data from , which is subject to data use agreements.

To reproduce the full analysis:

* Apply for access via: https://adni.loni.usc.edu/
* Follow preprocessing steps described in the paper

---

## Citation

If you use this code, please cite our preprint:

Kondo, Y., Honda, N., for the Alzheimer's Disease Neuroimaging Initiative.  
Spatially resolved mapping of tau amplification rates via differentiable simulation of prion-like propagation.  
bioRxiv, 2026.  
https://www.biorxiv.org/content/10.64898/2026.06.02.729568v1

```bibtex
@article{kondo2026differentiabletau,
  title = {Spatially resolved mapping of tau amplification rates via differentiable simulation of prion-like propagation},
  author = {Kondo, Yohei and Honda, Naoki and {{for the Alzheimer's Disease Neuroimaging Initiative}}},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.06.02.729568},
  url = {https://www.biorxiv.org/content/10.64898/2026.06.02.729568v1}
}
```
---

## Acknowledgements

This work builds on publicly available neuroimaging tools and datasets, including FSL, FreeSurfer, the Human Connectome Project, the Allen Human Brain Atlas, and the Alzheimer's Disease Neuroimaging Initiative (ADNI).

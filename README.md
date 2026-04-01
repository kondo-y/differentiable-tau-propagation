# Differentiable Tau Propagation (Minimal Release)

This repository provides a **minimal, self-contained implementation** of the core computational framework introduced in our preprint:

> *Spatially resolved mapping of tau amplification rates via differentiable simulation of prion-like propagation*

---

## Overview

This repository focuses on methodological contributions of the paper:

* Differentiable reaction–diffusion simulation of prion-like propagation
* MRI-informed model parametrization
* GPU-accelerated implementation using JAX
* End-to-end differentiability for gradient-based optimization

Due to the use of restricted neuroimaging datasets (e.g., ADNI), reproducing the complete analysis pipeline requires controlled data access.
Instead, this repository exposes the core computational components.
We believe the transparency is useful for evaluating the methodological contribution of the work.

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

## What is NOT included (yet)

The following components depend on restricted datasets (e.g., ADNI) and are **not included in this release**:

* Tau PET data processing pipeline
* Subject-level parameter inference scripts
* Full training / optimization pipeline used in the paper

Some of these components will be released upon publication, subject to data sharing policies.

---

## Data availability

The original study uses data from , which is subject to data use agreements.

To reproduce the full analysis:

* Apply for access via: https://adni.loni.usc.edu/
* Follow preprocessing steps described in the paper

---

## Citation

(The preprint below is not posted yet)

If you use this code, please cite:

```
Kondo, Y., Naoki, H.
Spatially resolved mapping of tau amplification rates via differentiable simulation of prion-like propagation
(preprint)
```

---

## Acknowledgements

This work builds on publicly available neuroimaging tools and datasets, including FSL, FreeSurfer, the Human Connectome Project, the Allen Human Brain Atlas, and the Alzheimer's Disease Neuroimaging Initiative (ADNI).

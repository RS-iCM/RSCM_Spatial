# RSCM_Spatial: Remote sensing–driven crop growth simulation and yield forecastin

**Authors**: Chi Tim Ng at Hang Seng University of Hong Kong and Jonghan Ko at Chonnam National University

**Collaborator**: Seungtaek Jeong at Korea Aerospace Institute and Jong-oh Ban at Hallym Polytechnic University

**Repository for the model**: https://github.com/RS-iCM/RSCM_Spatial

**Repository for bigger data**: https://huggingface.co/datasets/jonghanko/RSCM_Spatial/tree/main

---

## Overview

RSCM_Spatial is an open-source Python framework for simulating crop growth and yield from field to regional scales. By integrating satellite-derived vegetation indices (VIs) and climate data into automated parameterization, it reduces reliance on ground calibration while supporting flexible LAI regression methods (empirical, Bayesian, and machine learning). The platform includes pretrained models, reproducible notebooks, and built-in visualization, enabling scalable applications across crops, environments, and management systems.

---

## Features

- Seamless integration of remote sensing and climate data for crop modeling
- Multiple LAI regression frameworks: empirical, Bayesian log–log, and machine learning
- Extendable API for incorporating custom vegetation indices and algorithms
- Pretrained models and reproducible Jupyter notebooks for rice, wheat, and maize
- Supports 1D (time-series) and 2D (geospatial) simulation workflows
- Built-in visualization tools for time-series dynamics, scatter diagnostics, and spatial mapping
- Modular architecture for straightforward adaptation to additional crops and regions

---

## Requirements

- Python ≥ 3.10  
- numpy  
- pandas  
- matplotlib  
- scikit-learn
- cartopy
- geopandas
- rasterio
- shapely
- scipy
- seaborn

Install dependencies using:

```bash
pip install -r requirements.txt


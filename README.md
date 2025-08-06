# GMM-Based Analysis of Mosquito Swarm Structure

This repository contains code for analyzing 3D flight data of *Anopheles coluzzii* mosquitoes using Gaussian Mixture Models (GMMs). The pipeline includes data preprocessing, ellipsoid fitting, model fitting, and visualization to explore how swarm structure varies across female:male group ratios.

## Overview

The analysis pipeline performs the following steps:

1. **Data Import and Time Correction**  
   - Loads multiple replicate files containing mosquito 3D trajectories.  
   - Applies time correction based on recording metadata.

2. **Preprocessing and Filtering**  
   - Filters for biologically meaningful flight speeds and spatial zones.  
   - Restricts analysis to a defined 20-minute swarming period.  
   - Filters short-duration tracks.

3. **Ellipsoid Fitting**  
   - Uses GMMs to fit a 3D ellipsoid to each replicate swarm.  
   - Computes ellipsoid volume and identifies points within 90% confidence bounds.

4. **Speed–Volume Modeling**  
   - Extracts mean flight speed within each ellipsoid.  
   - Fits linear, nonlinear, and spline models to assess relationships.

5. **Visualization**  
   - Generates 3D plots of swarms and ellipsoids.  
   - Plots residuals, KDEs, boxplots, and fitted curves.

6. **Export**  
   - Saves the data points inside the fitted ellipsoids for further analysis.

## Requirements

- Python ≥ 3.8  
- Dependencies (install via `pip` or `conda`):
  ```bash
  pandas numpy matplotlib seaborn scipy scikit-learn
Data Availability
All datasets are available via Open Science Framework (OSF):
🔗 https://osf.io/6nkyq/

Download and place all .csv files into a local data/ directory before running the analysis.

## Data Description

All datasets in the `data/` folder are anonymized and publicly available.

### Example Variables:
- `track_id`: unique identifier for each mosquito track  
- `time`: timestamp (seconds)  
- `x, y, z`: 3D position coordinates (mm)  
- `sex`: biological sex (M/F)  
- `group`: experimental OSR treatment (e.g., M-only, 3:1 M:F)  
- `volume`: ellipsoid volume (mm³)  
- `mean_speed`: average mosquito speed in replicate (mm/s)

---

## Reproducing Results

For instructions on reproducing the manuscript figures, please refer to [USAGE.md](USAGE.md). This includes commands and step-by-step guidance on:

- Fitting ellipsoids to swarm replicates
- Estimating volumes and computing speed metrics
- Fitting statistical models to explore volume-speed relationships
  
Download the data from OSF and place in a data/ folder.

Run the scripts

## Please note this is a work in progress and adjustments are constantly made. If you encounter any errors please contact me at svielma@itg.be

# 3D Swarm Structure and Speed Analysis and Gaussian Model *Anopheles* Mosquitoes

This repository contains the code used to process, analyze, and visualize 3D mosquito flight data to quantify swarm structure and its relationship with mean speed. It includes time correction, filtering, GMM-based ellipsoid fitting, volume estimation, and model fitting of volume-speed relationships.

## Overview

This pipeline was developed as part of the thesis *Behavioral Ecology and Spatial Dynamics of Anopheles coluzzii Swarms* and is specifically tailored to process experimental datasets examining the effect of varying female group compositions on swarm shape and internal flight speed dynamics.

Key outputs include:
- Cleaned and filtered 3D mosquito tracks
- GMM-fitted ellipsoids for each swarm replicate
- Calculated ellipsoid volumes and mean speeds
- Model fits (linear, quadratic, cubic, logistic, spline)

## Dependencies

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- scipy  


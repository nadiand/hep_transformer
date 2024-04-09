# Translating Hits to Tracks: Transforming High Energy Physics Experiments

This repository contains the implementation of the Transformer Regressor: an encoder-only model for track finding. It is a sequence-to-sequence model, taking a sequence of hits from a single event as input, and producing a sequence of track parameters: one per hit, characterizing the track of the particle that generated each hit. In a secondary step, a clustering algorithm is used to group hits belonging to the same particle together (HDBSCAN).

The repository implements the whole pipeline, from data loading, to training, to evaluation. The scoring of the model is done using the TrackML score, which is taken from the trackML github page (https://github.com/LAL/trackml-library/tree/master), and three efficiency metrics defined in the GNN Tracking project (https://github.com/gnn-tracking/gnn_tracking/tree/main).

The paper describing the experiments done and results obtained with this code base is also included in the repository.

## Dependencies
The code runs on Python==3.9.7 and uses torch==2.1.2!

Other libraries used are as follows: numpy, pandas, matplotlib, hdbscan.

## Contents
The Transformer Regressor architecture is implemented in `model.py`, and the Flash attention and custom encoder layer using it are implemented in `custom_encoder.py`. The `training.py` file contains the training functionality when exact attention is used, while `training_flash.py` contains training funcitonality with Flash attention enabled. 

The `evaluation\` directory contains functionality used at inference time: `test.py` for obtaining scores and heatmaps; `performance_stats.py` for obtaining MSE, CUDA and CPU time, standard deviation; `plotting.py` for the creation of the heatmaps; `scoring.py` for the TrackML and efficiency calculations.

The `data_processing\` directory contains functionality related to the simulated data: `dataset.py` for loading REDVID data, and a data class for the hits and their associated particle IDs; `trackml_data.py` for loading the TrackML data and transforming the TrackML events into smaller ones for the creation of the subset datasets. The `domain_decomposition.py` functionality is not used in the reported experiments but is fully functional. 

The `refiner/` folder contains the implementation of a refiner network (training, testing), which is not used in the reported experiments but is also fully functional.

The trained models for which best scores are reported in the paper, are included in the `models/` folder.

## Author
All code, unless explicitly stated, is produced by Nadezhda Dobreva.
The author is part of the wider Trackformers: Transformers for tracking project. This repository contains only the implementations contributed to Nadezhda Dobreva.
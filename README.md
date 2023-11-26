# Encoder-only Transformer Regressor for Hit Classification in High Energy Physics

This repository contains the implementation of a transformer encoder-only model for trajectory reconstruction, its training, testing, and scoring. It is a sequence-to-sequence model, taking a sequence of hits (from a single event) and producing a sequence of track parameters (one for each hit), characterizing the track of the particle that generated each hit. In a secondary step, a clustering algorithm is used to group hits belonging to the same particle together (Agglomerative Clustering).

The scoring of the model is done using the TrackML score, which is taken from the trackML github page (https://github.com/LAL/trackml-library/tree/master).
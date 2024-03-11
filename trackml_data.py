import torch
import numpy as np
import pandas as pd
import random
import argparse

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from model import PAD_TOKEN
from domain_decomposition import split_event

def transform_trackml_data(event_id, split=False):
    hits_data = pd.read_csv(f'event0000{event_id}-hits.csv')
    particles_data = pd.read_csv(f'event0000{event_id}-particles.csv')
    truth_data = pd.read_csv(f'event0000{event_id}-truth.csv')

    # Merge hit, truth and particle dataframes into a single one with the relevant variables
    hits_data = hits_data[["hit_id", "x", "y", "z", "volume_id"]]
    hits_data.insert(0, column="particle_id", value=truth_data["particle_id"].values)
    hits_data.insert(0, column="weight", value=truth_data["weight"].values)

    merged_data = hits_data.merge(truth_data, left_on='hit_id', right_on='hit_id')
    merged_data = merged_data.merge(particles_data, left_on='particle_id_x', right_on='particle_id')

    final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
    final_data['event_id'] = event_id

    # Split up the event into multiple subevents, using domain decomposition
    if split:
        split_data = split_event(final_data, int(event_id))
        final_data = split_data.sort_values('event_class')

    # Write the sub-events to a file
    data_type = random.choices(["train", "test", "val"], cum_weights=[70, 85, 100])[0]
    if data_type == "train":
        final_data.to_csv('trackml_train_data_subdivided.csv', mode='a', index=False, header=False)
    elif data_type == "test":
        final_data.to_csv('trackml_test_data_subdivided.csv', mode='a', index=False, header=False)
    else:
        final_data.to_csv('trackml_validation_data_subdivided.csv', mode='a', index=False, header=False)


def load_trackml_data(data, max_num_hits, normalize=False, chunking=False):
    if not chunking:
        data = pd.read_csv(data)

    # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event = shuffled_data.groupby("event_id")

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,0]**2 + event_track_params_data[:,1]**2 + event_track_params_data[:,2]**2)
        theta = np.arccos(event_track_params_data[:,2]/p)
        phi = np.arctan2(event_track_params_data[:,1], event_track_params_data[:,0])
        processed_event_track_params_data = np.column_stack([theta, np.sin(phi), event_track_params_data[:,3]])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data


def load_preprocessed_trackml_data(data_path, max_num_hits):
    data = pd.read_csv(data_path)

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event = shuffled_data.groupby("event_id")

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["theta","phi","q"]].to_numpy(dtype=np.float32)
        return np.pad(event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--event_id')
    args = argparser.parse_args()
    transform_trackml_data(args.event_id)

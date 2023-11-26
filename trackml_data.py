import torch
import numpy as np
import pandas as pd
import random
import argparse

from model import PAD_TOKEN

def transform_trackml_data(event_id, min_part_in_event, max_part_in_event):
    hits_data = pd.read_csv(f'event0000{event_id}-hits.csv')
    particles_data = pd.read_csv(f'event0000{event_id}-particles.csv')
    truth_data = pd.read_csv(f'event0000{event_id}-truth.csv')

    # get only X many particles' data from the event, with X in [min_part_in_event, max_part_in_event]
    nr_particles_in_event = random.randint(min_part_in_event, max_part_in_event)
    sampled_particle_ids = particles_data['particle_id'].unique().tolist()[:nr_particles_in_event]
    indices = particles_data['particle_id'].isin(sampled_particle_ids)
    particles_data = particles_data[indices]

    indices = truth_data['particle_id'].isin(particles_data["particle_id"])
    truth_data = truth_data[indices]
    hits_data = hits_data[indices]

    # merge dataframes into a single one
    hits_data = hits_data[["hit_id", "x", "y", "z", "volume_id"]]
    hits_data.insert(0, column="particle_id", value=truth_data["particle_id"].values)
    hits_data.insert(0, column="weight", value=truth_data["weight"].values)

    merged_data = hits_data.merge(truth_data, left_on='hit_id', right_on='hit_id')
    merged_data = merged_data.merge(particles_data, left_on='particle_id_x', right_on='particle_id')

    final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
    final_data['event_id'] = event_id

    # write it to a file
    final_data.to_csv(f'trackml_{min_part_in_event}to{max_part_in_event}tracks.csv', mode='a', index=False, header=False)


def load_trackml_data(data_path, max_num_hits, normalize=False):
    full_data = pd.read_csv(data_path)

    if normalize:
        for col in ["x", "y", "z", "vx", "vy", "vz", "px", "py", "pz", "q"]:
            mean = full_data[col].mean()
            std = full_data[col].std()
            full_data[col] = (full_data[col] - mean)/std

    full_data = full_data.sample(frac=1)
    data_grouped_by_event = full_data.groupby("event_id")

    def extract_hits_data(event_rows):
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        event_track_params_data = event_rows[["vx","vy","vz","px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,3]**2 + event_track_params_data[:,4]**2 + event_track_params_data[:,5]**2)
        theta = np.arccos(event_track_params_data[:,5]/p)
        phi = np.arctan2(event_track_params_data[:,4], event_track_params_data[:,3])
        processed_event_track_params_data = np.column_stack([theta, phi, event_track_params_data[:,6]])
        return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, track_params_data, hit_classes_data

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--event_id')
    argparser.add_argument('-l', '--min_nr_tracks')
    argparser.add_argument('-u', '--max_nr_tracks')
    args = argparser.parse_args()
    transform_trackml_data(args.event_id, args.min_nr_tracks, args.max_nr_tracks)
    
    # transform_trackml_data('21000', 1, 10)

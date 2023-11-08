import torch
import numpy as np
import pandas as pd
import argparse

from model import PAD_TOKEN

def transform_trackml_data(event_id):
    hits_data = pd.read_csv(f'event0000{event_id}-hits.csv')
    particles_data = pd.read_csv(f'event0000{event_id}-particles.csv')
    truth_data = pd.read_csv(f'event0000{event_id}-truth.csv')

    # take the data corresponding to only 50 particles
    sampled_particle_ids = particles_data['particle_id'].unique().tolist()[:50]
    indices = particles_data['particle_id'].isin(sampled_particle_ids)
    particles_data = particles_data[indices]

    indices = truth_data['particle_id'].isin(particles_data["particle_id"])
    truth_data = truth_data[indices]
    hits_data = hits_data[indices]

    # take only those hits that are in the inner detector range
    indices = hits_data['volume_id'] <= 9
    hits_data = hits_data[indices]
    truth_data = truth_data[indices]

    indices = particles_data['particle_id'].isin(truth_data['particle_id'])
    particles_data = particles_data[indices]

    # merge dataframes into a single one
    hits_data = hits_data[["hit_id", "x", "y", "z", "volume_id"]]
    hits_data.insert(0, column="particle_id", value=truth_data["particle_id"].values)
    hits_data.insert(0, column="weight", value=truth_data["weight"].values)

    merged_data = hits_data.merge(truth_data, left_on='hit_id', right_on='hit_id')
    merged_data = merged_data.merge(particles_data, left_on='particle_id_x', right_on='particle_id')

    final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
    final_data['event_id'] = event_id

    # write it to a file
    final_data.to_csv('data.csv', mode='a', index=False, header=False)

def load_trackml_data(data_path, max_num_hits):
    full_data = pd.read_csv(data_path)
    full_data = full_data.sample(frac=1)

    data_grouped_by_event = full_data.groupby("event_id")

    def extract_hits_data(event_rows):
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float16)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        event_track_params_data = event_rows[["vx","vy","vz","px","py","pz","q"]].to_numpy(dtype=np.float16)
        return np.pad(event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float16)
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
    args = argparser.parse_args()
    transform_trackml_data(args.event_id)
    
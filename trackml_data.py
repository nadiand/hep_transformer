import torch
import numpy as np
import pandas as pd
import random
import argparse

from scoring import score_event
from model import PAD_TOKEN

# TODO: make the bins overlap!
def split_data(file_name):
    data = pd.read_csv(file_name)

    # Calculate theta and phi of each hit
    p = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['theta'] = np.arccos(data['z']/p)
    data['phi'] = np.arctan2(data['y'], data['x'])
    
    # n_bins is per parameter; so overall there will be n_bins x n_bins many
    n_bins = 3
    theta_bin_size = ((data['theta'].max()-data['theta'].min()) / n_bins) + 1e-7
    phi_bin_size = ((data['phi'].max()-data['phi'].min()) / n_bins) + 1e-7
    data['theta_bin'] = ((data['theta'] - data['theta'].min()) // theta_bin_size).astype(int)
    data['phi_bin'] = ((data['phi'] - data['phi'].min()) // phi_bin_size).astype(int)

    # leaving 0 empty for when there's padding eventually (TODO is this necessary?)
    data['subset_id'] = 1 + (data['theta_bin'] * n_bins + data['phi_bin'])

    # Modify event ID so that it is split up into multiple smaller events
    event_ids = []
    for ind, val in enumerate(data['event_id'].values):
        event_ids.append(val+data['subset_id'][ind])

    data['event_id'] = event_ids

    # Evaluating using trackML (poor evaluator since we don't want a 1-to-1 mapping between
    # subevents and tracks)
    data.insert(0, column='hit_id', value=np.arange(0, len(data)))
    truth = pd.DataFrame(data[['hit_id', 'particle_id', 'weight']])
    truth.columns = ['hit_id', 'particle_id', 'weight']
    submission = pd.DataFrame(data[['hit_id', 'subset_id']])
    submission.columns = ['hit_id', 'track_id']
    score = score_event(truth, submission)
    print(score)

    # Evaluating using split up events
    # Find the track IDs placed in each subset of the data
    subset_ids = data['subset_id'].unique().tolist()
    particle_ids_per_subset = []
    for sub_id in subset_ids:
        indices = data['subset_id'] == sub_id
        hits = data[indices]
        particle_ids_per_subset.append(hits['particle_id'].unique().tolist())
    
    # Keep track of which particles occur in multiple subsets
    overlapping_particles = []
    for i in range(1, len(particle_ids_per_subset)-1):
        overlapping_particles.append(list(set(particle_ids_per_subset[i]) & set(particle_ids_per_subset[i-1])))
    print(overlapping_particles)
    
    data.to_csv(f'trackml_subdivided.csv', mode='a', index=False, header=False)


def transform_trackml_data(event_id, min_part_in_event, max_part_in_event, sub_events):
    hits_data = pd.read_csv(f'event0000{event_id}-hits.csv')
    particles_data = pd.read_csv(f'event0000{event_id}-particles.csv')
    truth_data = pd.read_csv(f'event0000{event_id}-truth.csv')

    # Find the particle IDs of this event
    particle_ids = particles_data['particle_id'].unique().tolist()
    nr_particles_in_current_event = 0
    for i in range(int(sub_events)):
        # get only X many particles' data from the event, with X in [min_part_in_event, max_part_in_event]
        nr_particles_in_event = random.randint(int(min_part_in_event), int(max_part_in_event))
        sampled_particle_ids = particle_ids[nr_particles_in_current_event:nr_particles_in_current_event+nr_particles_in_event]
        nr_particles_in_current_event += nr_particles_in_event
        indices = particles_data['particle_id'].isin(sampled_particle_ids)
        particles_data_subset = particles_data[indices]

        indices = truth_data['particle_id'].isin(particles_data_subset["particle_id"])
        truth_data_subset = truth_data[indices]
        hits_data_subset = hits_data[indices]

        # Merge hit, truth and particle dataframes into a single one with the relevant variables
        hits_data_subset = hits_data_subset[["hit_id", "x", "y", "z", "volume_id"]]
        hits_data_subset.insert(0, column="particle_id", value=truth_data_subset["particle_id"].values)
        hits_data_subset.insert(0, column="weight", value=truth_data_subset["weight"].values)

        merged_data = hits_data_subset.merge(truth_data_subset, left_on='hit_id', right_on='hit_id')
        merged_data = merged_data.merge(particles_data_subset, left_on='particle_id_x', right_on='particle_id')

        final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
        # Make sure the subevents don't have the same event ID
        final_data['event_id'] = event_id + f"_{i}"

        # Write the event(s) to a file
        final_data.to_csv(f'trackml_{min_part_in_event}to{max_part_in_event}tracks.csv', mode='a', index=False, header=False)


def load_trackml_data(data_path, max_num_hits, normalize=False):
    full_data = pd.read_csv(data_path)

    # Normalizing the data if applicable
    if normalize:
        for col in ["x", "y", "z", "vx", "vy", "vz", "px", "py", "pz", "q"]:
            mean = full_data[col].mean()
            std = full_data[col].std()
            full_data[col] = (full_data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    full_data = full_data.sample(frac=1)
    data_grouped_by_event = full_data.groupby("event_id")

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["vx","vy","vz","px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,3]**2 + event_track_params_data[:,4]**2 + event_track_params_data[:,5]**2)
        theta = np.arccos(event_track_params_data[:,5]/p)
        phi = np.arctan2(event_track_params_data[:,4], event_track_params_data[:,3])
        processed_event_track_params_data = np.column_stack([theta, phi, event_track_params_data[:,6]])
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--event_id')
    argparser.add_argument('-l', '--min_nr_tracks')
    argparser.add_argument('-u', '--max_nr_tracks')
    argparser.add_argument('-s', '--nr_subevents')
    args = argparser.parse_args()
    transform_trackml_data(args.event_id, args.min_nr_tracks, args.max_nr_tracks, args.nr_subevents)
    
    # Equivalent to not splitting the event into multiple ones
    # transform_trackml_data(event_id='21000', min_part_in_event=1, max_part_in_event=5, sub_events=1)


import torch
import numpy as np
import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from model import PAD_TOKEN


def make_bins(data, parameter, bin_edges, overlap):
    num_bins = len(bin_edges) - 1
    bin_names = [f'{parameter}_bin{i}' for i in range(1, num_bins + 1)]

    for i in range(num_bins):
        lower_threshold = bin_edges[i] - overlap
        upper_threshold = bin_edges[i + 1] + overlap
        if i == 0 and parameter == 'phi':
            data[bin_names[i]] = np.logical_or(data[parameter] < upper_threshold, data[parameter] > np.pi - overlap)
        elif i == num_bins-1 and parameter == 'phi':
            data[bin_names[i]] = np.logical_or(data[parameter] >= lower_threshold, data[parameter] < -np.pi + overlap)
        else:
            data[bin_names[i]] = np.logical_and(data[parameter] >= lower_threshold, data[parameter] < upper_threshold)


def split_event(data, event_id):
    overlap_theta = overlap_phi = 0.

    # Calculate theta and phi of each hit
    p = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['theta'] = np.arccos(data['z']/p)
    data['phi'] = np.arctan2(data['y'], data['x'])

    # Create the bins of theta (values currently chosen based on distribution of theta)
    # theta_bin_edges = np.array([0, 0.5, 2.5, np.pi]) # 3 bins
    theta_bin_edges = np.array([0, 0.3, 1.45, 2.8, np.pi]) # 4 bins
    make_bins(data, 'theta', theta_bin_edges, overlap_theta)

    # Create the bins of phi (values currently chosen based on distribution of phi)
    # phi_bin_edges = np.array([-np.pi, -1, 1, np.pi]) # 3 bins
    phi_bin_edges = np.array([-np.pi, -1.57, 0, 1.57, np.pi]) # 4 bins
    make_bins(data, 'phi', phi_bin_edges, overlap_phi)

    # plt.hist(data['phi'].values, bins=[-np.pi, -1.57, 0, 1.57, np.pi])
    # plt.show()
    
    # Create the bins of theta-phi combinations
    classes = []
    for phi_bin in range(1, len(phi_bin_edges)):
        for theta_bin in range(1, len(theta_bin_edges)):
            class_name = f'class_{phi_bin}_{theta_bin}'
            phi_condition = data[f'phi_bin{phi_bin}']
            theta_condition = data[f'theta_bin{theta_bin}']
            data[class_name] = np.logical_and(phi_condition, theta_condition)
            classes.append(class_name)

    # For every row in the dataset, check which "classes" it got assigned to
    # and add all of them to a new list of rows. Necessary step since the overlap
    # might lead to the same hit belonging in e.g. 3 different "classes" and in 
    # that case we want that row to be duplicated with a different event ID 3 times
    event_class = []
    new_rows = []
    for i, class_name in enumerate(classes):
        for _, row in data[data[class_name]].iterrows():
            event_class.append(f"{event_id}_{i}")
            new_rows.append(row)

    # Create the new dataframe with the newly composed lists
    new_data = pd.DataFrame(new_rows, columns=data.columns)
    new_data['event_class'] = event_class
    print(new_data['event_class'].value_counts())

    # Evaluate the split by calculating its "efficiency" score
    evaluate_split_event(data, new_data)

    # Important: event_class is now the new event_id to follow for separation 
    # between events in other scripts!
    return new_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x", "event_id", "event_class"]] 

def evaluate_split_event(old_data, data):
    # Make a dictionary with all particle_ids and the number of hits they have
    all_particle_ids = old_data['particle_id'].unique().tolist()
    particle_dict = {}
    for part_id in all_particle_ids:
        indices = old_data['particle_id'] == part_id
        particle_dict[part_id] = len(old_data[indices])

    # For every track/particle, calculate the ratio that shows how well represented 
    # it is by the class it's in: (hits of that particle in this class)/(all hits for that particle).
    # Where we go over each class and each particle, so that if a particle is in multiple
    # classes, we take the one it's best represented in
    class_ids = data['event_class'].unique().tolist()
    portions = {}
    for cl_id in class_ids:
        indices = data['event_class'] == cl_id
        hits_data = data[indices]
        particle_ids = hits_data['particle_id'].unique().tolist()
        for part_id in particle_ids:
            indices = hits_data['particle_id'] == part_id
            nr_hits_in_class = len(hits_data[indices])
            if part_id not in portions.keys():
                portions[part_id] = nr_hits_in_class/particle_dict[part_id]
            else:
                if nr_hits_in_class/particle_dict[part_id] > portions[part_id]:
                    portions[part_id] = nr_hits_in_class/particle_dict[part_id]

    portions_arr = np.array(list(portions.values()))
    print(portions_arr)
    print("Average efficiency:", portions_arr.mean())
    print("Efficiency standard dev:", portions_arr.std())


def transform_trackml_data(event_id):
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
    split_data = split_event(final_data, int(event_id))
    ready_data = split_data.sort_values('event_class')

    # Write the sub-events to a file
    # data_type = random.choices(["train", "test", "val"], cum_weights=[70, 15, 15])[0]
    # if data_type == "train":
    #     ready_data.to_csv('trackml_train_data_subdivided.csv', mode='a', index=False, header=False)
    # elif data_type == "test":
    #     ready_data.to_csv('trackml_test_data_subdivided.csv', mode='a', index=False, header=False)
    # else:
    #     ready_data.to_csv('trackml_validation_data_subdivided.csv', mode='a', index=False, header=False)


def load_trackml_data(data_path, normalize=False):
    data = pd.read_csv(data_path)
    # Find the max number of hits in the batch to pad up to
    # events = data['event_class'].unique()
    # event_lens = [len(data[data['event_class'] == event]) for event in events]
    events = data['event_id'].unique()
    event_lens = [len(data[data['event_id'] == event]) for event in events]
    max_num_hits = max(event_lens)

    # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "vx", "vy", "vz", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event = shuffled_data.groupby("event_id")
    # data_grouped_by_event = data.groupby("event_class")

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        event_track_params_data = event_rows[["vx","vy","vz","px","py","pz","q"]].to_numpy(dtype=np.float32)
        p = np.sqrt(event_track_params_data[:,3]**2 + event_track_params_data[:,4]**2 + event_track_params_data[:,5]**2)
        theta = np.arccos(event_track_params_data[:,5]/p)
        ita = -np.log(np.tan(theta/2))
        # normalized_ita = np.exp(ita) / (1 + np.exp(ita))
        phi = np.arctan2(event_track_params_data[:,4], event_track_params_data[:,3])
        processed_event_track_params_data = np.column_stack([ita, phi, event_track_params_data[:,6]])
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
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('-e', '--event_id')
    # args = argparser.parse_args()
    # transform_trackml_data(args.event_id)
    
    transform_trackml_data(event_id='21000')

    # rows = {'theta' : [0.3, 0.7], 'phi': [0.2, 0.4], 'particle_id': [1, 1]}
    # df = pd.DataFrame(rows)
    # new_df = split_event(df, 21000)
    
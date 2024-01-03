import torch
import numpy as np
import pandas as pd
import random
import argparse

from model import PAD_TOKEN

def split_event(data, event_id):
    # print(data)
    overlap = 0.2

    # Calculate theta and phi of each hit
    p = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['theta'] = np.arccos(data['z']/p)
    data['phi'] = np.arctan2(data['y'], data['x'])

    data['theta_bin1'] = data['theta'] < 0.5 + overlap
    data['theta_bin2'] = np.logical_and(data['theta'] >= 0.5 - overlap, data['theta'] < 2.5 + overlap)
    data['theta_bin3'] = data['theta'] >= 2.5 - overlap

    data['phi_bin1'] = np.logical_or(data['phi'] < -1 + overlap, data['phi'] > np.pi - overlap)
    data['phi_bin2'] = np.logical_and(data['phi'] >= -1 - overlap, data['phi'] < 1 + overlap)
    data['phi_bin3'] = np.logical_or(data['phi'] >= 1 - overlap, data['phi'] < -np.pi + overlap)

    data['class1'] = np.logical_and(data['phi_bin1'], data['theta_bin1'])
    data['class2'] = np.logical_and(data['phi_bin2'], data['theta_bin1'])
    data['class3'] = np.logical_and(data['phi_bin3'], data['theta_bin1'])
    data['class4'] = np.logical_and(data['phi_bin1'], data['theta_bin2'])
    data['class5'] = np.logical_and(data['phi_bin2'], data['theta_bin2'])
    data['class6'] = np.logical_and(data['phi_bin3'], data['theta_bin2'])
    data['class7'] = np.logical_and(data['phi_bin1'], data['theta_bin3'])
    data['class8'] = np.logical_and(data['phi_bin2'], data['theta_bin3'])
    data['class9'] = np.logical_and(data['phi_bin3'], data['theta_bin3'])

    event_class = []
    new_rows = []
    for _, row in data.iterrows():
        if row['class1']:
            event_class.append(f"{event_id}_1")
            new_rows.append(row)
        if row['class2']:
            event_class.append(f"{event_id}_2")
            new_rows.append(row)
        if row['class3']:
            event_class.append(f"{event_id}_3")
            new_rows.append(row)
        if row['class4']:
            event_class.append(f"{event_id}_4")
            new_rows.append(row)
        if row['class5']:
            event_class.append(f"{event_id}_5")
            new_rows.append(row)
        if row['class6']:
            event_class.append(f"{event_id}_6")
            new_rows.append(row)
        if row['class7']:
            event_class.append(f"{event_id}_7")
            new_rows.append(row)
        if row['class8']:
            event_class.append(f"{event_id}_8")
            new_rows.append(row)
        if row['class9']:
            event_class.append(f"{event_id}_9")
            new_rows.append(row)

    new_data = pd.DataFrame(new_rows, columns=data.columns)
    new_data['event_class'] = event_class

    # print(new_data[["x", "y", "z", "theta", "phi", "event_class", "particle_id"]])

    evaluate_split_event(data, new_data)

    # event_class is now the new event_id to follow for separation between events!
    return new_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x", "event_id", "event_class"]] 

def evaluate_split_event(old_data, data):
    # Make a dictionary with all particle_ids and the number of hits they have
    all_particle_ids = old_data['particle_id'].unique().tolist()
    particle_dict = {}
    for part_id in all_particle_ids:
        indices = old_data['particle_id'] == part_id
        particle_dict[part_id] = len(old_data[indices])

    portions = {}
    # Check for each event class, for each particle in it, how many hits of that particle
    # are in the event class and find the ratio (hits of that particle in this class)/(all hits for that particle)
    class_ids = data['event_class'].unique().tolist()
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


def load_trackml_data(data, normalize=False):
    # Find the max number of hits in the batch to pad up to
    events = data['event_class'].unique()
    event_lens = [len(data[data['event_class'] == event]) for event in events]
    # events = data['event_id'].unique()
    # event_lens = [len(data[data['event_id'] == event]) for event in events]
    max_num_hits = max(event_lens)

    # Normalize the data if applicable
    if normalize:
        for col in ["x", "y", "z", "vx", "vy", "vz", "px", "py", "pz", "q"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    data = data.sample(frac=1)
    # data_grouped_by_event = data.groupby("event_id")
    data_grouped_by_event = data.groupby("event_class")

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
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('-e', '--event_id')
    # args = argparser.parse_args()
    # transform_trackml_data(args.event_id)
    
    transform_trackml_data(event_id='21000')

    # rows = {'theta' : [0.3, 0.7], 'phi': [0.2, 0.4], 'particle_id': [1, 1]}
    # df = pd.DataFrame(rows)
    # new_df = split_event(df, 21000)
    
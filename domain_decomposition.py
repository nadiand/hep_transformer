# import torch
import numpy as np
import pandas as pd
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib
from joblib import Parallel, delayed
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

PAD_TOKEN = -1


def split_event(data, overlap_theta=0.1, overlap_phi=0.1, num_bins_theta=5, num_bins_phi=5):
    # Calculate theta and phi of each hit
    p = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['theta'] = np.arccos(data['z']/p)
    data['phi'] = np.arctan2(data['y'], data['x'])

    # store theta and phi max and min values before normalization
    theta_max = data['theta'].max()
    theta_min = data['theta'].min()
    phi_max = data['phi'].max()
    phi_min = data['phi'].min()

    # normalize theta and phi to be between 0 and 1
    data['theta'] = (data['theta'] - data['theta'].min())/(data['theta'].max() - data['theta'].min())
    data['phi'] = (data['phi'] - data['phi'].min())/(data['phi'].max() - data['phi'].min())

    # find cut values that equally, with overlap_theta overlap, subdivide the data into bins of theta
    theta_bins = []
    theta_bins.append((0, data['theta'].quantile(1/num_bins_theta + overlap_theta)))
    for i in range(1, num_bins_theta-1):
        theta_bins.append((data['theta'].quantile(i/num_bins_theta - overlap_theta/2), 
                           data['theta'].quantile((i+1)/num_bins_theta + overlap_theta/2)))
    theta_bins.append((data['theta'].quantile((num_bins_theta-1)/num_bins_theta - overlap_theta), 1))

    # find cut values that equally, with overlap_phi overlap, subdivide the data into bins of phi
    phi_bins = []
    phi_bins.append((0, data['phi'].quantile(1/num_bins_phi + overlap_phi)))
    for i in range(1, num_bins_phi-1):
        phi_bins.append((data['phi'].quantile(i/num_bins_phi - overlap_phi/2), data['phi'].quantile((i+1)/num_bins_phi + overlap_phi/2)))
    phi_bins.append((data['phi'].quantile((num_bins_phi-1)/num_bins_phi - overlap_phi), 1))


    data_subdivided = pd.DataFrame(columns=data.columns)
    for i in range(num_bins_phi):
        for j in range(num_bins_theta):
            indices = ((data["theta"] > theta_bins[j][0]) & (data["theta"] < theta_bins[j][1]) 
                       & (data["phi"] > phi_bins[i][0]) & (data["phi"] < phi_bins[i][1]))
            data_class = data[indices]
            data_class["event_class"] = i*num_bins_phi + j
            data_subdivided = data_subdivided.append(data_class)

    return data_subdivided, data, theta_bins, phi_bins

def split_event_fixedbins(data, num_bins_theta=5, num_bins_phi=5, theta_bins=None, phi_bins=None):

    # Calculate theta and phi of each hit
    p = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['theta'] = np.arccos(data['z']/p)
    data['phi'] = np.arctan2(data['y'], data['x'])

    # store theta and phi max and min values before normalization
    theta_max = data['theta'].max()
    theta_min = data['theta'].min()
    phi_max = data['phi'].max()
    phi_min = data['phi'].min()

    # normalize theta and phi to be between 0 and 1
    data['theta'] = (data['theta'] - data['theta'].min())/(data['theta'].max() - data['theta'].min())
    data['phi'] = (data['phi'] - data['phi'].min())/(data['phi'].max() - data['phi'].min())

    data_subdivided = pd.DataFrame(columns=data.columns)
    for i in range(num_bins_phi):
        for j in range(num_bins_theta):
            indices = ((data["theta"] > theta_bins[j][0]) & (data["theta"] < theta_bins[j][1]) 
                       & (data["phi"] > phi_bins[i][0]) & (data["phi"] < phi_bins[i][1]))
            data_class = data[indices]
            data_class["event_class"] = i*num_bins_phi + j
            data_subdivided = data_subdivided.append(data_class)

    return data_subdivided, data

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
    average_size = 0
    portions = {}
    for cl_id in class_ids:
        indices = data['event_class'] == cl_id
        hits_data = data[indices]
        average_size += len(hits_data)
        particle_ids = hits_data['particle_id'].unique().tolist()
        for part_id in particle_ids:
            indices = hits_data['particle_id'] == part_id
            nr_hits_in_class = len(hits_data[indices])
            if part_id not in portions.keys():
                portions[part_id] = nr_hits_in_class/particle_dict[part_id]
            else:
                if nr_hits_in_class/particle_dict[part_id] > portions[part_id]:
                    portions[part_id] = nr_hits_in_class/particle_dict[part_id]
    average_size /= len(class_ids)


    portions_arr = np.array(list(portions.values()))
    print(portions_arr)
    return portions_arr.mean(), portions_arr.std(), average_size
    print("Average efficiency:", portions_arr.mean())
    print("Efficiency standard dev:", portions_arr.std())


def transform_trackml_data(event_id, overlap_theta=0.1, overlap_phi=0.1, num_bins_theta=5, num_bins_phi=5, theta_bins=None, phi_bins=None):
    hits_data = pd.read_csv(f'../../data/event0000{event_id}-hits.csv')
    particles_data = pd.read_csv(f'../../data/event0000{event_id}-particles.csv')
    truth_data = pd.read_csv(f'../../data/event0000{event_id}-truth.csv')

    # Merge hit, truth and particle dataframes into a single one with the relevant variables
    hits_data = hits_data[["hit_id", "x", "y", "z", "volume_id"]]
    hits_data.insert(0, column="particle_id", value=truth_data["particle_id"].values)
    hits_data.insert(0, column="weight", value=truth_data["weight"].values)

    merged_data = hits_data.merge(truth_data, left_on='hit_id', right_on='hit_id')
    merged_data = merged_data.merge(particles_data, left_on='particle_id_x', right_on='particle_id')

    final_data = merged_data[["x", "y", "z", "volume_id", "vx", "vy", "vz", "px", "py", "pz", "q", "particle_id", "weight_x"]]
    final_data.loc[:, 'event_id'] = np.repeat(event_id, len(final_data))
    # Split up the event into multiple subevents, using domain decomposition
    if event_id == '21000':
        data_subdivided, data, theta_bins, phi_bins = split_event(final_data, overlap_theta=overlap_theta, overlap_phi=overlap_phi, num_bins_theta=num_bins_theta, num_bins_phi=num_bins_phi)
    else:
        data_subdivided, data = split_event_fixedbins(final_data, num_bins_theta=num_bins_theta, num_bins_phi=num_bins_phi, theta_bins=theta_bins, phi_bins=phi_bins)
    # ready_data = split_data.sort_values('event_class')

    # Write the sub-events to a file
    # data_type = random.choices(["train", "test", "val"], cum_weights=[70, 15, 15])[0]
    # if data_type == "train":
    #     ready_data.to_csv('trackml_train_data_subdivided.csv', mode='a', index=False, header=False)
    # elif data_type == "test":
    #     ready_data.to_csv('trackml_test_data_subdivided.csv', mode='a', index=False, header=False)
    # else:
    #     ready_data.to_csv('trackml_validation_data_subdivided.csv', mode='a', index=False, header=False)

    return data_subdivided, data, theta_bins, phi_bins


# def load_trackml_data(data, normalize=False):
#     # Find the max number of hits in the batch to pad up to
#     # events = data['event_class'].unique()
#     # event_lens = [len(data[data['event_class'] == event]) for event in events]
#     events = data['event_id'].unique()
#     event_lens = [len(data[data['event_id'] == event]) for event in events]
#     max_num_hits = max(event_lens)

#     # Normalize the data if applicable
#     if normalize:
#         for col in ["x", "y", "z", "vx", "vy", "vz", "px", "py", "pz", "q"]:
#             mean = data[col].mean()
#             std = data[col].std()
#             data[col] = (data[col] - mean)/std

#     # Shuffling the data and grouping by event ID
#     data = data.sample(frac=1)
#     data_grouped_by_event = data.groupby("event_id")
#     # data_grouped_by_event = data.groupby("event_class")

#     def extract_hits_data(event_rows):
#         # Returns the hit coordinates as a padded sequence; this is the input to the transformer
#         event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
#         return np.pad(event_hit_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

#     def extract_track_params_data(event_rows):
#         # Returns the track parameters as a padded sequence; this is what the transformer must regress
#         event_track_params_data = event_rows[["vx","vy","vz","px","py","pz","q"]].to_numpy(dtype=np.float32)
#         p = np.sqrt(event_track_params_data[:,3]**2 + event_track_params_data[:,4]**2 + event_track_params_data[:,5]**2)
#         theta = np.arccos(event_track_params_data[:,5]/p)
#         phi = np.arctan2(event_track_params_data[:,4], event_track_params_data[:,3])
#         processed_event_track_params_data = np.column_stack([theta, phi, event_track_params_data[:,6]])
#         return np.pad(processed_event_track_params_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

#     def extract_hit_classes_data(event_rows):
#         # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
#         event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)
#         return np.pad(event_hit_classes_data, [(0, max_num_hits-len(event_rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

#     # Get the hits, track params and their weights as sequences padded up to a max length
#     grouped_hits_data = data_grouped_by_event.apply(extract_hits_data)
#     grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
#     grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

#     # Stack them together into one tensor
#     hits_data = torch.tensor(np.stack(grouped_hits_data.values))
#     track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
#     hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

#     return hits_data, track_params_data, hit_classes_data

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--event_id')
    args = argparser.parse_args()

    # evaluate the efficiency score as a function of the overlap and the number of bins, and store in 2d matrix
    overlaps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    num_bins = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    efficiency_matrix = np.zeros((len(overlaps), len(num_bins)))
    efficiency_std_matrix = np.zeros((len(overlaps), len(num_bins)))
    average_size_matrix = np.zeros((len(overlaps), len(num_bins)))

    # def evaluate_split_event_wrapper(overlap, num_bins):
    #     try:
    #         data_subdivided, data = transform_trackml_data(event_id=event_id, overlap_theta=overlap, overlap_phi=overlap, num_bins_theta=num_bins, num_bins_phi=num_bins)
    #         return evaluate_split_event(data, data_subdivided)
    #     except ValueError:
    #         return 0, 0, 0

    bins_dict = {}
    theta_bins = []
    phi_bins = []
    breaking = False
    for i in range(len(overlaps)):
        # run the wrapper in parallel
        # results = Parallel(n_jobs=-1)(delayed(evaluate_split_event_wrapper)(overlaps[i], num_bins[j]) for j in range(len(num_bins)))
        for j in range(len(num_bins)):
            try:
                data_subdivided, data, theta_bins, phi_bins = transform_trackml_data(event_id=args.event_id, overlap_theta=overlaps[i], overlap_phi=overlaps[i], num_bins_theta=num_bins[j], num_bins_phi=num_bins[j], theta_bins=theta_bins, phi_bins=phi_bins)
                results = evaluate_split_event(data, data_subdivided)
                bins_dict[(i,j)] = (theta_bins, phi_bins)
                breaking = False
            except ValueError:
                breaking = True

            if breaking:
                break
            
            if args.event_id != '21000':
                # store results in matrix
                efficiency_matrix[i,j] = results[0]
                efficiency_std_matrix[i,j] = results[1]
                average_size_matrix[i,j] = results[2]

    if args.event_id == '21000':
        with open('bins.txt', 'a') as file:
            file.write(json.dumps(bins_dict))
        exit(0)

    efficiency_matrix = np.ma.masked_where(efficiency_matrix < 0.05, efficiency_matrix)
    efficiency_std_matrix = np.ma.masked_where(efficiency_matrix < 0.05, efficiency_std_matrix)
    average_size_matrix = np.ma.masked_where(efficiency_matrix < 0.05, average_size_matrix)

    cmap = matplotlib.cm.get_cmap("OrRd").copy()

    cmap.set_bad(color='black')

    # plot efficiency matrix
    fig, ax = plt.subplots()
    im = ax.imshow(efficiency_matrix, cmap=cmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(num_bins)))
    ax.set_yticks(np.arange(len(overlaps)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(num_bins)
    ax.set_yticklabels(overlaps)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(overlaps)):
        for j in range(len(num_bins)):
            text = ax.text(j, i, np.round(efficiency_matrix[i, j], 4),
                        ha="center", va="center", color="w")
            
    ax.set_title("Efficiency score as a function of overlap and number of bins")
    fig.tight_layout()
    # plt.show()

    # save image to file
    fig.savefig(f'efficiency_score_{args.event_id}.png')

    # plot number of values in each bin
    fig, ax = plt.subplots()
    im = ax.imshow(average_size_matrix, cmap=cmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(num_bins)))
    ax.set_yticks(np.arange(len(overlaps)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(num_bins)
    ax.set_yticklabels(overlaps)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(overlaps)):
        for j in range(len(num_bins)):
            text = ax.text(j, i, np.round(average_size_matrix[i, j], 0), fontsize="small",
                        ha="center", va="center", color="w")
    
    ax.set_title(f"Average number of hits in each bin for event {args.event_id}")
    fig.tight_layout()
    # plt.show()

    # save image to file
    fig.savefig(f'average_number_hits_{args.event_id}.png')
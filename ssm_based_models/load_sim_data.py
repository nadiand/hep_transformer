import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_TOKEN = 0

class HitsDataset(Dataset):
    '''
    Dataset class for the detector data, i.e. the hit coordinates, their track
    parameters, the particles they belong to.
    '''

    def __init__(self, hits_data, track_params_data=None, class_data=None, hits_seq_len=None):
        self.hits_data = hits_data.to(DEVICE)
        self.track_params_data = track_params_data.to(DEVICE)
        self.class_data = class_data.to(DEVICE)
        self.hits_seq_len = hits_seq_len.to(DEVICE)
        self.total_events = self.__len__()

    def __len__(self):
        return self.hits_data.shape[0]

    def __getitem__(self, idx):
        return idx, self.hits_data[idx], self.hits_seq_len[idx], self.track_params_data[idx], self.class_data[idx]


def get_dataloaders(dataset, train_frac, valid_frac, test_frac, batch_size):
    train_set, valid_set, test_set = random_split(dataset, [train_frac, valid_frac, test_frac], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_trackml_data(data, normalize=True, sort=False, spherical_system=False, cylindrical_system=False):
    '''
    Function for reading .csv file with TrackML data and creating tensors
    containing the hits and ground truth information from it.
    max_num_hits denotes the size of the largest event, to pad the other events
    up to. normalize decides whether the data will be normalized first.
    chunking allows for reading .csv files in chunks.
    '''
    data = pd.read_csv(data)

    print("nr rows", len(data), flush=True)
    print("nr events", data["event_id"].nunique(), flush=True)
    print(data.dtypes)
    print(data.isnull().sum())
    print("Any infs:", np.isinf(data.select_dtypes(include=[float, int])).any().any())

    if normalize:
        for col in ["x", "y", "z"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1, random_state=37)
    data_grouped_by_event = shuffled_data.groupby("event_id")
    max_num_hits = data_grouped_by_event.size().max() + 1
    # Round up to the next multiple of 128 for flex attention
    max_num_hits = ((max_num_hits + 127) // 128) * 128

    def sort_on_distance(event_coords):
        distances = []
        for p in event_coords:
            distances.append((p[0]**2 + p[1]**2 + p[2]**2)**0.5)

        order = np.argsort(distances)
        return order

    def sort_side_to_side(event_coords):
        order = np.lexsort((event_coords[:,2], event_coords[:,1], event_coords[:,0]))
        return order

    def spherical_coord(event_hit_data):
        r = np.sqrt(event_hit_data[:,0]**2 + event_hit_data[:,1]**2 + event_hit_data[:,2]**2)
        phi = np.arctan2(event_hit_data[:,1], event_hit_data[:,0])
        theta = np.arccos(event_hit_data[:,2]/r)
        if normalize:
            r_norm = r/4.727182 # r_max = 4.727182
            phi_norm = (phi + np.pi)/(2*np.pi)
            theta_norm = theta/np.pi
            new_event_hit_data = np.column_stack([r_norm, theta_norm, phi_norm])
        else:
            new_event_hit_data = np.column_stack([r, theta, phi])

        # order = np.lexsort((new_event_hit_data[:,0], new_event_hit_data[:,2], new_event_hit_data[:,1]))
        order = circular_sort_phi(new_event_hit_data[:,2])
        new_event_hit_data = np.column_stack([r_norm, theta_norm, np.sin(phi_norm), np.cos(phi_norm)])
        return new_event_hit_data, order

    def circular_sort_phi(phi_values):
        # Compute mean direction (not just mean, to handle wraparound properly)
        mean_angle = np.arctan2(np.mean(np.sin(phi_values)), np.mean(np.cos(phi_values)))

        # Rotate so that mean_angle is at 0
        phi_shifted = (phi_values - mean_angle + np.pi) % (2*np.pi) - np.pi

        # Get sorting order
        order = np.argsort(phi_shifted)
        return order

    def cylindrical_coord(event_hit_data):
        rho = np.sqrt(event_hit_data[:,0]**2 + event_hit_data[:,1]**2)
        phi = np.arctan2(event_hit_data[:,1], event_hit_data[:,0])
        z = event_hit_data[:,2]
        new_event_hit_data = np.column_stack([rho, phi, z])
        if normalize:
            rho_norm = rho/3.855715 # rho_max = 3.855715
            phi_norm = (phi + np.pi) / (2 * np.pi)
            z_norm = (z + 2.774814) / (2 * 2.774814) # Z_max = 2.774814
            new_event_hit_data = np.column_stack([rho_norm, phi_norm, z_norm])

        order = np.lexsort((new_event_hit_data[:,0], new_event_hit_data[:,2], new_event_hit_data[:,1]))
        return new_event_hit_data, order

    def extract_hits_data(event_rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        sequence_length = len(event_rows)
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)

        if spherical_system:
            event_hit_data, order = spherical_coord(event_hit_data)
            if sort:
                event_hit_data = event_hit_data[order]

        elif cylindrical_system:
            event_hit_data, order = cylindrical_coord(event_hit_data)
            if sort:
                event_hit_data = event_hit_data[order]

        elif sort:
            # order = sort_on_distance(event_hit_data)
            order = sort_side_to_side(event_hit_data)
            event_hit_data = event_hit_data[order]

        return np.pad(event_hit_data, [(0, max_num_hits-sequence_length), (0, 0)], "constant", constant_values=PAD_TOKEN), sequence_length

    def extract_track_params_data(event_rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        sequence_length = len(event_rows)
        event_track_params_data = event_rows[["cos_phi","sin_phi","cos_theta","q"]].to_numpy(dtype=np.float32)

        if sort:
            event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
            if spherical_system:
                _, order = spherical_coord(event_hit_data)
            elif cylindrical_system:
                _, order = cylindrical_coord(event_hit_data)
            else:
                # order = sort_on_distance(event_hit_data)
                order = sort_side_to_side(event_hit_data)
            event_track_params_data = event_track_params_data[order]

        return np.pad(event_track_params_data, [(0, max_num_hits-sequence_length), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_hit_classes_data(event_rows):
        # Returns the particle information as a padded sequence; this is used for weighting in the calculation of trackML score
        sequence_length = len(event_rows)
        event_hit_classes_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)

        if sort:
            event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
            if spherical_system:
                _, order = spherical_coord(event_hit_data)
            elif cylindrical_system:
                _, order = cylindrical_coord(event_hit_data)
            else:
                # order = sort_on_distance(event_hit_data)
                order = sort_side_to_side(event_hit_data)
            event_hit_classes_data = event_hit_classes_data[order]

        return np.pad(event_hit_classes_data, [(0, max_num_hits-sequence_length), (0, 0)], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    results = data_grouped_by_event.apply(extract_hits_data)
    grouped_hits_data, sequence_lengths = zip(*results)
    grouped_track_params_data = data_grouped_by_event.apply(extract_track_params_data)
    grouped_hit_classes_data = data_grouped_by_event.apply(extract_hit_classes_data)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data))
    hits_data_seq_lengths = torch.tensor(sequence_lengths, dtype=torch.long)
    track_params_data = torch.tensor(np.stack(grouped_track_params_data.values))
    hit_classes_data = torch.tensor(np.stack(grouped_hit_classes_data.values))

    return hits_data, hits_data_seq_lengths, track_params_data, hit_classes_data

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_TOKEN = 0
CLASSES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
C = len(CLASSES)
N = 1320
S = N*C
O = 300

class HitsDataset(Dataset):
    '''
    Dataset class for the detector data, i.e. the hit coordinates, their track
    parameters, the particles they belong to.
    '''

    def __init__(self, hits_data, track_params_data=None, class_data=None, hits_class_info=None):
        self.hits_data = hits_data.to(DEVICE)
        self.track_params_data = track_params_data.to(DEVICE)
        self.class_data = class_data.to(DEVICE)
        self.hits_class_info = hits_class_info.to(DEVICE)
        self.total_events = self.__len__()

    def __len__(self):
        return self.hits_data.shape[0]

    def __getitem__(self, idx):
        return idx, self.hits_data[idx], self.hits_class_info[idx], self.track_params_data[idx], self.class_data[idx]


def get_dataloaders(dataset, train_frac, valid_frac, test_frac, batch_size):
    train_set, valid_set, test_set = random_split(dataset, [train_frac, valid_frac, test_frac], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_trackml_data(data, normalize=True):
    data = pd.read_csv(data)

    if normalize:
        for col in ["x", "y", "z"]:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean)/std

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1, random_state=37)
    data_grouped_by_event = shuffled_data.groupby("event_id")

    def extract_event_data(event_rows):
        event_hit_data = event_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        event_param_data = event_rows[["cos_phi","sin_phi","cos_theta","q"]].to_numpy(dtype=np.float32)
        event_weight_data = event_rows[["particle_id","weight"]].to_numpy(dtype=np.float32)

        # Convert cartesian to spherical coords and normalize (optional)
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

        # Sort by phi only
        # print('HITS', event_weight_data)
        order = np.argsort(new_event_hit_data[:,2])
        # print(order)
        sorted_hits = new_event_hit_data[order]
        sorted_params = event_param_data[order]
        sorted_weights = event_weight_data[order]
        # print('SORTED', sorted_weights)

        # Go over every defined class and create a new list containing the hits from that class followed by
        # padding (until the max nr hits per class have been reached). Also make a list containing 0s and 1s
        # that will be used to generate the attention mask to not attend to padding
        all_class_info, new_coords, new_params, new_weights = [], [], [], []
        ind = 0
        for c in range(C):
            hit = sorted_hits[ind]
            class_info = []
            while hit[2] >= CLASSES[c][0] and (hit[2] < CLASSES[c][1] or (hit[2] == 1.0 and c == 4)):
                class_info.append(1)
                new_coords.append(hit)
                new_params.append(sorted_params[ind])
                new_weights.append(sorted_weights[ind])
                ind += 1
                if ind >= len(sorted_hits):
                    break
                hit = sorted_hits[ind]
            remaining = N - len(class_info)
            class_info.extend([0]*remaining)
            all_class_info.extend(class_info)
            new_coords.extend([[PAD_TOKEN]*3]*remaining)
            new_params.extend([[PAD_TOKEN]*4]*remaining)
            new_weights.extend([[PAD_TOKEN]*2]*remaining)
            if ind >= len(sorted_hits):
                break
        if sum(class_info) == 0:
            print("very bad")
            
        # Pad up even more, in case we stopped early (i.e. event only has hits in the first few classes)
        # Shouldn't be needed but as a sanity check
        total_needed = S - len(new_coords)
        if total_needed > 0:
            new_coords.extend([[PAD_TOKEN]*3]*total_needed)
            new_params.extend([[PAD_TOKEN]*4]*total_needed)
            new_weights.extend([[PAD_TOKEN]*2]*total_needed)
            all_class_info.extend([0]*total_needed)

        # print('NEW', new_weights)
        # print('INFO', all_class_info)
        # Make sure the length of the event is the max seq len S
        assert len(new_coords) == S
        return np.array(new_coords, dtype=np.float32), np.array(all_class_info, dtype=np.int32), np.array(new_params, dtype=np.float32), np.array(new_weights, dtype=np.float32)

    grouped_hits_data = []
    class_infos = []
    grouped_params = []
    grouped_weights = []

    for _, event_rows in data_grouped_by_event:
        hits, cls, params, weights = extract_event_data(event_rows)
        grouped_hits_data.append(hits)
        grouped_params.append(params)
        grouped_weights.append(weights)
        class_infos.append(cls)

    hits_data = torch.tensor(np.stack(grouped_hits_data), dtype=torch.float32)
    hits_class_data = torch.tensor(np.stack(class_infos), dtype=torch.bool)
    track_params_data = torch.tensor(np.stack(grouped_params))
    hit_classes_data = torch.tensor(np.stack(grouped_weights))
    return hits_data, hits_class_data, track_params_data, hit_classes_data


def build_fixed_class_mask():
    # TODO: currently hardcoded for 5 classes! fix later on
    hit_class_matrix = torch.zeros((S, C))
    # assign hits to classes
    hit_class_matrix[:N+O, 0] = 1
    hit_class_matrix[N:N*2+O, 1] = 1
    hit_class_matrix[N*2:N*3+O, 2] = 1
    hit_class_matrix[N*3:N*4+O, 3] = 1
    hit_class_matrix[N*4:, 4] = 1
    # and add the wraparound corners
    hit_class_matrix[(N*5-O):, 0] = 1
    hit_class_matrix[:O, 4] = 1

    # shared[i, j] = number of classes shared between hits i and j
    shared = hit_class_matrix @ hit_class_matrix.T
    mask = shared > 0
    return mask

def build_full_mask_fn(mask, padding_loc):
    """
    Takes the precomputed mask and adds the padding mask onto it.
    """
    def mask_generation(b, h, q_idx, kv_idx):
        paddings = padding_loc[b] # 0/1 depending on whether at this location in the sequence we have a padding
        pad = paddings[q_idx] & paddings[kv_idx]
        return mask[q_idx, kv_idx] & pad
    return mask_generation

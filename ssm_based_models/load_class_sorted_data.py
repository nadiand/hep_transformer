import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_TOKEN = 0
CLASSES = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
OVERLAP = 0.05
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
        all_class_info, all_new_coords, all_new_params, all_new_weights = [], [], [], []
        ind = 0
        for c in range(C):
            hit = sorted_hits[ind]
            insert_pad = -1
            class_info, new_coords, new_params, new_weights = [], [], [], []
            class_ind = 0
            while hit[2] >= CLASSES[c][0] and (hit[2] < CLASSES[c][1] or (hit[2] == 1.0 and c == (C-1))):
                if hit[2] >= (CLASSES[c][1] - OVERLAP):
                    insert_pad = class_ind
                class_info.append(1)
                new_coords.append(hit)
                new_params.append(sorted_params[ind])
                new_weights.append(sorted_weights[ind])
                ind += 1
                class_ind += 1
                if ind >= len(sorted_hits):
                    break
                hit = sorted_hits[ind]

            remaining = N - len(class_info)
            if insert_pad == 0:
                class_info = [0]*remaining + class_info
                new_coords = [[PAD_TOKEN]*3]*remaining + new_coords
                new_params = [[PAD_TOKEN]*4]*remaining + new_params
                new_weights = [[PAD_TOKEN]*2]*remaining + new_weights
            elif insert_pad == -1:
                class_info.extend([0]*remaining)
                new_coords.extend([[PAD_TOKEN]*3]*remaining)
                new_params.extend([[PAD_TOKEN]*4]*remaining)
                new_weights.extend([[PAD_TOKEN]*2]*remaining)
            else:
                class_info = class_info[:insert_pad] + [0]*remaining + class_info[insert_pad:]
                new_coords = new_coords[:insert_pad] + [[PAD_TOKEN]*3]*remaining + new_coords[insert_pad:]
                new_params = new_params[:insert_pad] + [[PAD_TOKEN]*4]*remaining + new_params[insert_pad:]
                new_weights = new_weights[:insert_pad] + [[PAD_TOKEN]*2]*remaining + new_weights[insert_pad:]

            all_class_info.extend(class_info)
            all_new_coords.extend(new_coords)
            all_new_params.extend(new_params)
            all_new_weights.extend(new_weights)
            if ind >= len(sorted_hits):
                break

        # Pad up even more, in case we stopped early (i.e. event only has hits in the first few classes)
        total_needed = S - len(all_new_coords)
        if total_needed > 0:
            all_new_coords.extend([[PAD_TOKEN]*3]*total_needed)
            all_new_params.extend([[PAD_TOKEN]*4]*total_needed)
            all_new_weights.extend([[PAD_TOKEN]*2]*total_needed)
            all_class_info.extend([0]*total_needed)

        # print('NEW', new_coords)
        # print('INFO', all_class_info)
        # Make sure the length of the event is the max seq len S
        assert len(all_new_coords) == S
#        print(all_class_info)
        return np.array(all_new_coords, dtype=np.float32), np.array(all_class_info, dtype=np.int32), np.array(all_new_params, dtype=np.float32), np.array(all_new_weights, dtype=np.float32)
    
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
    hit_class_matrix = torch.zeros((S, C))
    # assign to class
    hit_class_matrix[:N, 0] = 1
    for i in range(1, C-1):
        hit_class_matrix[N*i:N*(i+1), i] = 1
    hit_class_matrix[N*(C-1):, (C-1)] = 1

    # shared[i, j] = number of classes shared between hits i and j
    shared = hit_class_matrix @ hit_class_matrix.T
    mask = shared > 0

    # add overlap between classes
    for i in range(C - 1):
        class_i_end = N*(i+1)
        class_j_start = N*(i+1)

        i_overlap = slice(class_i_end-O, class_i_end)
        j_overlap = slice(class_j_start, class_j_start+O)

        mask[i_overlap, j_overlap] = True
        mask[j_overlap, i_overlap] = True

    # circular wraparound overlap (phi periodic continuity)
    i_overlap = slice(N*(C-1) + (N-O), N*C)
    j_overlap = slice(0, O)
    mask[i_overlap, j_overlap] = True
    mask[j_overlap, i_overlap] = True
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

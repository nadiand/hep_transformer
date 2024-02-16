import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from collections import Counter

from model import PAD_TOKEN


class ClustersDataset(Dataset):

    def __init__(self, hit_clusters, labels):
        self.hit_clusters = hit_clusters
        self.labels = labels
        self.total_clusters = self.__len__()

    def __len__(self):
        return self.hit_clusters.shape[0]

    def __getitem__(self, idx):
        return idx, self.hit_clusters[idx], self.labels[idx]
    

def get_dataloaders(dataset, train_frac, valid_frac, batch_size):
    train_set, valid_set = random_split(dataset, [train_frac, valid_frac], generator=torch.Generator().manual_seed(37))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def load_calibration_data(data_path):
    data = pd.read_csv(data_path)

    # Shuffling the data and grouping by event ID
    shuffled_data = data.sample(frac=1)
    data_grouped_by_event_particle = shuffled_data.groupby(['event_id', 'cluster_id'])
    max_num_hits = data_grouped_by_event_particle.size().reset_index().max()[0]

    def extract_input(rows):
        # Returns the hit coordinates as a padded sequence; this is the input to the transformer
        hit_data = rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return np.pad(hit_data, [(0, max_num_hits-len(rows)), (0, 0)], "constant", constant_values=PAD_TOKEN)

    def extract_output(rows):
        # Returns the track parameters as a padded sequence; this is what the transformer must regress
        label_data = rows["particle_id"].to_numpy(dtype=np.float32)
        data = Counter(label_data)
        majority_label = data.most_common(1)[0][0]
        track_belonging = np.array([label == majority_label for label in label_data], dtype=int)
        return np.pad(track_belonging, [(0, max_num_hits-len(rows))], "constant", constant_values=PAD_TOKEN)

    # Get the hits, track params and their weights as sequences padded up to a max length
    grouped_hits_data = data_grouped_by_event_particle.apply(extract_input)
    grouped_track_belonging_data = data_grouped_by_event_particle.apply(extract_output)

    # Stack them together into one tensor
    hits_data = torch.tensor(np.stack(grouped_hits_data.values))
    belonging_data = torch.tensor(np.stack(grouped_track_belonging_data.values))

    return hits_data, belonging_data
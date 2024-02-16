import torch
import numpy as np
import pandas as pd
from collections import Counter

from model import PAD_TOKEN


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
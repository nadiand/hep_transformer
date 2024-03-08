import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np
import time

from training import clustering
from scoring import calc_score, calc_score_trackml
from model import TransformerClassifier, PAD_TOKEN
from dataset import HitsDataset, get_dataloaders, load_curved_3d_data, load_linear_3d_data, load_linear_2d_data
from trackml_data import load_trackml_data
from plotting import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_with_mse(model, test_loader):
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}

    # Transformer physics performance
    loss_fn = nn.MSELoss()
    pred_true_differences = []

    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        difference = loss_fn(pred, track_params)
        pred_true_differences.append(difference.item())

    print(f'Avg MSE: {sum(pred_true_differences)/len(pred_true_differences)}')
    print(f'Std MSE: {np.std(pred_true_differences)}')

    return predictions


def predict_with_cudatime(model, test_loader, min_cl_size, min_samples):
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()

    # Time performance
    start_event = cuda.Event(enable_timing=True)
    end_event = cuda.Event(enable_timing=True)

    i = 0
    for data in test_loader:
        _, hits, track_params, track_labels = data
        if i > 0:
            start_event.record()

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        cluster_labels = clustering(pred, min_cl_size, min_samples)

        i += 1
    end_event.record()
    cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("CUDA time:", elapsed_time_ms/len(test_loader))


transformer = TransformerClassifier(num_encoder_layers=6,
                                    d_model=32,
                                    n_head=8,
                                    input_size=2,
                                    output_size=1,
                                    dim_feedforward=128,
                                    dropout=0.1)
transformer = transformer.to(DEVICE)

checkpoint = torch.load("2d_model_best", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_linear_2d_data(data_path="hits_and_tracks_2d_events_all.csv") #, max_num_hits=100)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')

min_cl_size, min_samples = 5, 5
preds = predict_with_cudatime(transformer, test_loader, min_cl_size, min_samples)
preds = list(preds.values())


# for param in ['theta', 'phi', 'q']:
#     plot_heatmap(preds, param, "test")
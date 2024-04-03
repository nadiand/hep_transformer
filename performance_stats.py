import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np
import time

from time import perf_counter_ns, process_time_ns

from training import clustering
from scoring import calc_score, calc_score_trackml
from model import TransformerClassifier, PAD_TOKEN
from dataset import HitsDataset, get_dataloaders, load_curved_3d_data, load_linear_3d_data, load_linear_2d_data
from trackml_data import load_trackml_data
from plotting import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_with_stats(model, test_loader, min_cl_size, min_samples):
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.

    # Transformer physics performance
    loss_fn = nn.MSELoss()
    pred_true_differences = []

    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Make prediction
        hits = hits
        track_params = track_params
        track_labels = track_labels

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        difference = loss_fn(pred, track_params)
        pred_true_differences.append(difference.item())

        cluster_labels = clustering(pred, min_cl_size, min_samples)

        event_score, scores = calc_score_trackml(cluster_labels[0], track_labels[0])
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)

    print(f'Avg MSE: {sum(pred_true_differences)/len(pred_true_differences)}')
    print(f'Std MSE: {np.std(pred_true_differences)}')
    print(f'TrackML score: {score/len(test_loader)}')
    print(perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader))

    return predictions


def record_cuda_cpu_time(model, test_loader, min_cl_size, min_samples):
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()

    # Time performance
    cuda_times = []
    cpu_times = []
    start_event = cuda.Event(enable_timing=True)
    end_event = cuda.Event(enable_timing=True)

    for data in test_loader:
        _, hits, track_params, track_labels = data
        start_event.record()

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)
        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)
        
        end_event.record()
        cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        cuda_times.append(elapsed_time_ms)

        start_cpu_time = process_time_ns()
        cluster_labels = clustering(pred, min_cl_size, min_samples)
        end_cpu_time = process_time_ns()
        cpu_times.append(end_cpu_time - start_cpu_time)

    print("Avg CUDA time:", sum(cuda_times[1:])/len(cuda_times[1:]))
    print("Avg CPU time:", sum(cpu_times[1:])/len(cpu_times[1:]))
    

transformer = TransformerClassifier(num_encoder_layers=6,
                                    d_model=32,
                                    n_head=4,
                                    input_size=3,
                                    output_size=4,
                                    dim_feedforward=128,
                                    dropout=0.1)
transformer = transformer.to(DEVICE)

checkpoint = torch.load("models/10to50_sin_cos_best", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_trackml_data(data="trackml_10to50tracks_40kevents.csv", max_num_hits=700, normalize=True)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')

min_cl_size, min_samples = 5, 2
preds = record_cuda_cpu_time(transformer, test_loader, min_cl_size, min_samples)

# preds = list(preds.values())
# for param in ['theta', 'phi', 'q']:
#     plot_heatmap(preds, param, "test")

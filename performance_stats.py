import torch
import torch.nn as nn
import numpy as np
import time

from training import clustering
from scoring import calc_score, calc_score_trackml
from model import TransformerClassifier, PAD_TOKEN
from dataset import HitsDataset, get_dataloaders, load_curved_3d_data, load_linear_3d_data, load_linear_2d_data
from trackml_data import load_trackml_data
from plotting import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_with_stats(model, test_loader, dist_thresh):
    '''
    Evaluates the network on the test data. Returns the predictions
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score = 0.

    # Time performance
    prediction_times = []
    clustering_times = []

    # Transformer physics performance
    loss_fn = nn.MSELoss()
    pred_true_differences = []

    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Make prediction
        hits = hits.to(DEVICE)
        track_params = track_params.to(DEVICE)
        track_labels = track_labels.to(DEVICE)

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        before_model = time.time()
        pred = model(hits, padding_mask)
        after_model = time.time()
        prediction_times.append(after_model-before_model)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        difference = loss_fn(pred, track_params)
        pred_true_differences.append(difference.item())

        before_clustering = time.time()
        cluster_labels = clustering(pred, dist_thresh=dist_thresh)
        after_clustering = time.time()
        clustering_times.append(after_clustering-before_clustering)

        event_score = calc_score(cluster_labels[0], track_labels[0])
        score += event_score

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)

    print(f'Avg prediction time: {sum(prediction_times)/len(prediction_times)}')
    print(f'Std prediction time: {np.std(prediction_times)}')
    print(f'Avg clustering time: {sum(clustering_times)/len(clustering_times)}')
    print(f'Std clustering time: {np.std(clustering_times)}')
    print(f'Avg MSE: {sum(pred_true_differences)/len(pred_true_differences)}')
    print(f'Std MSE: {np.std(pred_true_differences)}')
    print(f'TrackML score: {score/len(test_loader)}')

    return predictions


transformer = TransformerClassifier(num_encoder_layers=6,
                                    d_model=32,
                                    n_head=8,
                                    input_size=3,
                                    output_size=3,
                                    dim_feedforward=128,
                                    dropout=0.1)
transformer = transformer.to(DEVICE)

checkpoint = torch.load("test_best", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_curved_3d_data(data_path="hits_and_tracks_3d_3curved_events_all.csv", max_num_hits=600)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')

preds = predict_with_stats(transformer, test_loader, dist_thresh=0.1)
preds = list(preds.values())


for param in ['theta', 'phi', 'q']:
    plot_heatmap(preds, param, "test")
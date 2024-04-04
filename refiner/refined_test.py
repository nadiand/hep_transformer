import torch
import pandas as pd

from ..model import TransformerRegressor
from ..dataset import HitsDataset, get_dataloaders, PAD_TOKEN
from ..plotting import plot_heatmap
from ..scoring import calc_score_trackml
from ..training import clustering
from training_refiner import refine
from refining_clusters_dataset import load_data_for_refiner

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_with_refined_clusters(regressor, test_loader, refiner, min_cl_size, min_samples):
    # Get the regressor in evaluation mode
    torch.set_grad_enabled(False)
    regressor.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.

    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Use regressor to make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = regressor(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)
        hits = torch.unsqueeze(hits[~padding_mask], 0)

        # Cluster track parameters
        cluster_labels = clustering(pred, min_cl_size, min_samples)

        # Create dataframe from cluster data to provide the refiner with
        rows = []
        for i in range(len(hits[0])):
            rows.append([hits[0][i][0].item(), hits[0][i][1].item(), hits[0][i][2].item(), track_params[0][i][0].item(), track_params[0][i][1].item(), track_params[0][i][2].item(),
                         track_params[0][i][3].item(), cluster_labels[0][i].item(), track_labels[0][i][0].item(), track_labels[0][i][1].item()])
        df = pd.DataFrame(rows, columns=['x','y','z','theta','sinphi','cosphi','q','cluster_id','track_id','weight'])
        
        # Use refiner to refine the clusters
        refiner_pred = refine(refiner, df)
        flattened_cluster_labels = [x for (xs,_) in refiner_pred for x in xs]
        flattened_track_ids = [x for (_,xs) in refiner_pred for x in xs]

        # Calculate scores based on refined clusters
        event_score, scores = calc_score_trackml(flattened_cluster_labels, flattened_track_ids)
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, flattened_cluster_labels, flattened_track_ids, event_score)

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)


transformer = TransformerRegressor(num_encoder_layers=6,
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

refiner = TransformerRegressor(num_encoder_layers=3,
                                    d_model=32,
                                    n_head=2,
                                    input_size=3,
                                    output_size=4,
                                    dim_feedforward=64,
                                    dropout=0.1)
refiner = refiner.to(DEVICE)
checkpoint = torch.load("refiner_best", map_location=torch.device('cpu'))
refiner.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_data_for_refiner(data="trackml_10to50tracks_40kevents.csv", normalize=True)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')


preds, score, perfect, double, lhc = predict_with_refined_clusters(transformer, test_loader, refiner, 5, 2)
print(score, perfect, double, lhc)
# preds = list(preds.values())
# for param in ["theta", "phi", "q"]:
#     plot_heatmap(preds, param, "test")
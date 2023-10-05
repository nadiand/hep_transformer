import torch
from model import TransformerClassifier
from training import predict
from dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from plotting import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = TransformerClassifier(num_encoder_layers=8,
                                    d_model=32,
                                    n_head=16,
                                    input_size=3,
                                    output_size=2,
                                    dim_feedforward=256,
                                    dropout=0.1)
transformer = transformer.to(DEVICE)
checkpoint = torch.load("3d_3curved_100epoch_big_best", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_curved_3d_data(data_path="hits_and_tracks_3d_3curved_events_all.csv", max_num_hits=540)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1024)
print('data loaded')

preds, score = predict(transformer, test_loader)
events = list(preds.keys())
preds = list(preds.values())
print(score)
# plot_3d(preds, "theta")
# plot_3d(preds, "phi")
# plot_cluster_labels(preds, events)
# plot_regressed_params(preds, events)
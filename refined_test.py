import torch

from model import TransformerClassifier
from training import predict, predict_with_refined_clusters
from dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from plotting import *
from trackml_data import load_trackml_data
from refiner_model import RefinerTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = TransformerClassifier(num_encoder_layers=6,
                                    d_model=32,
                                    n_head=4,
                                    input_size=3,
                                    output_size=3,
                                    dim_feedforward=128,
                                    dropout=0.1)
transformer = transformer.to(DEVICE)
checkpoint = torch.load("models/10to50_40k_sin_best", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

refiner = RefinerTransformer(num_encoder_layers=3,
                                    d_model=32,
                                    n_head=4,
                                    input_size=3,
                                    output_size=1,
                                    dim_feedforward=64,
                                    dropout=0.1)
refiner = refiner.to(DEVICE)
checkpoint = torch.load("refiner_best", map_location=torch.device('cpu'))
refiner.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_trackml_data(data="trackml_10to50tracks_40kevents.csv", max_num_hits=700, normalize=True)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')

# preds, score, _,_,_ = predict(transformer, test_loader, 5, 2)

preds, score = predict_with_refined_clusters(transformer, test_loader, refiner, 5, 2)
print(score)
preds = list(preds.values())
# for param in ["theta", "phi", "q"]:
#     plot_heatmap(preds, param, "test")
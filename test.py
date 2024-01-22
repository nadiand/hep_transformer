import torch
from model import TransformerClassifier
from training import predict
from dataset import HitsDataset, get_dataloaders
from plotting import *
from trackml_data import load_trackml_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = TransformerClassifier(num_encoder_layers=6,
                                    d_model=32,
                                    n_head=8,
                                    input_size=3,
                                    output_size=3,
                                    dim_feedforward=128,
                                    dropout=0.1)
transformer = transformer.to(DEVICE)
checkpoint = torch.load("test_last", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_trackml_data(data_path="trackml_data_50tracks.csv")
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')

preds, score = predict(transformer, test_loader)
print(score)
preds = list(preds.values())
for param in ["theta", "phi", "q"]:
    plot_heatmap(preds, param, "test")
import torch
from model import TransformerClassifier
from training import predict
from dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from plotting import *
from training import *
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(37)

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'test_score'
        },
    'parameters': {
        'num_enc_layers': {'min': 2, 'max': 8},
        'emb_dim': {'values': [8, 16, 32, 64, 128]},
        'nr_heads': {'values': [4, 8, 16, 32]},
        'dim_ff': {'values': [64, 128, 256]},
        'drop': {'values': [0.0, 0.1]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="hyperparam_optimization")

hits_data, track_params_data, track_classes_data = load_linear_3d_data(data_path="hits_and_tracks_3d_events_all.csv", max_num_hits=81) #540
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1024)

def train_one_model(num_enc_layers, emb_dim, nr_heads, dim_ff, drop):
    transformer = TransformerClassifier(num_encoder_layers=num_enc_layers,
                                                            d_model=emb_dim,
                                                            n_head=nr_heads,
                                                            input_size=3,
                                                            output_size=2,
                                                            dim_feedforward=dim_ff,
                                                            dropout=drop)
    transformer = transformer.to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

    for _ in range(50):
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn, True)
        val_loss = evaluate(transformer, valid_loader, loss_fn, True)
        wandb.log({"val_loss" : val_loss, "train_loss" : train_loss})

    _, score = predict(transformer, test_loader, True)
    wandb.log({"test_score": score})

def main():
    run = wandb.init()
    num_enc_layers = wandb.config.num_enc_layers
    emb_dim = wandb.config.emb_dim
    nr_heads = wandb.config.nr_heads
    dim_ff = wandb.config.dim_ff
    drop = wandb.config.drop
    if emb_dim % nr_heads == 0:
        for _ in range(3):
            train_one_model(num_enc_layers, emb_dim, nr_heads, dim_ff, drop)


wandb.agent(sweep_id, function=main)
wandb.finish()
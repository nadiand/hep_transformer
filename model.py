import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from trackml_data import load_trackml_data
from encoder_layer import MyEncoderLayer

class TransformerClassifier(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layers = MyEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, input, padding_mask):
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        out = self.decoder(memory)
        return out


def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.memory._record_memory_history(max_entries=100000)

    transformer = TransformerClassifier(num_encoder_layers=6,
                                            d_model=32,
                                            n_head=8,
                                            input_size=3,
                                            output_size=3,
                                            dim_feedforward=128,
                                            dropout=0.1)
    transformer = transformer.to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    hits_data, track_params_data, track_classes_data = load_trackml_data(data_path="../../trackml_data_700tracks.csv", normalize=True)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    train_loader, _, _ = get_dataloaders(dataset,
                                                                train_frac=0.7,
                                                                valid_frac=0.15,
                                                                test_frac=0.15,
                                                                batch_size=1)
    print('data loaded')

    def train_epoch(model, optim, train_loader, loss_fn, scaler):
        '''
        Conducts a single epoch of training: prediction, loss calculation, and loss
        backpropagation. Returns the average loss over the whole train data.
        '''
        # Get the network in train mode
        torch.set_grad_enabled(True)
        model.train()
        losses = 0.
        for data in train_loader:
            _, x, y, _ = data
            optim.zero_grad()

            # Make prediction
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.cuda.amp.autocast():
                padding_mask = (x == -1).all(dim=2)
                pred = model(x, padding_mask)
                # Calculate loss and use it to update weights
                loss = loss_fn(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses += loss.item()

        return losses / len(train_loader)

    train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn, scaler)
    print(train_loss)
    try:
        torch.cuda.memory._dump_snapshot("memory_usage.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    torch.cuda.memory._record_memory_history(enabled=None)

train()
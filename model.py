import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from trackml_data import load_trackml_data


class TransformerClassifier(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self):
        super(TransformerClassifier, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(3, 1, 4, 0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, 1)

    def forward(self, x):
        memory = self.encoder(src=x)
        return memory

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = TransformerClassifier()
transformer = transformer.to(DEVICE)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

max_nr_hits = 600
hits_data, track_params_data, track_classes_data = load_trackml_data(data_path="../../trackml_data_50tracks.csv")
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=32)
print('data loaded')

x = torch.Tensor([np.array([[-124,-85,-1502],[-107,-75,-1302],[-90,-64,-1102]])]) #, np.array([[-49.4727,-58.4256,-1298.0],[-41.6409,-49.6406,-1098.0],[-41.6409,-49.6406,-1098.0]])])
y = torch.Tensor([np.array([0.2])]) #, np.array([0.5])])

dummy_dataset = TensorDataset(x, y)
dummy_dataloader = DataLoader(dummy_dataset)

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
        x, y = data
        print(x)
        optim.zero_grad()

        # Make prediction
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.cuda.amp.autocast():
            pred = model(x)

            print(pred)

            # Calculate loss and use it to update weights
            loss = loss_fn(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        losses += loss.item()

    return losses / len(train_loader)

train_loss = train_epoch(transformer, optimizer, dummy_dataloader, loss_fn, scaler)
import torch
import torch.nn as nn
import numpy as np
# from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer

from encoder import *

PAD_TOKEN = -1

class AngleDifferenceLoss(nn.Module):
    def __init__(self):
        super(AngleDifferenceLoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((((predictions - targets) + np.pi) % 2*np.pi) - np.pi)

class TransformerClassifier(nn.Module):
    '''
    A transformer network for clustering hits that belong to the same trajectory.
    Takes the hits (i.e 2D or 3D coordinates) and outputs the probability of each
    hit belonging to each of the 20 possible tracks (classes).
    '''
    def __init__(self, num_encoder_layers, d_model, n_head, input_size, output_size, dim_feedforward, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, input, padding_mask):
        # print(input.shape)
        # print(input.dtype)
        x = self.input_layer(input)
        memory = self.encoder(src=x, src_key_padding_mask=padding_mask)
        out = self.decoder(memory)
        return out

def save_model(model, optim, type, val_losses, train_losses, epoch, count, file_name):
    print(f"Saving {type} model")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'count': count,
    }, f"{file_name}_{type}")
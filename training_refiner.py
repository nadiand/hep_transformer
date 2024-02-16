import torch
import torch.nn as nn
import numpy as np

from refiner_model import RefinerTransformer, PAD_TOKEN, save_model
from scoring import calc_score
from refining_clusters_dataloader import ClustersDataset, load_calibration_data, get_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, optim, train_loader, loss_fn):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.

    for data in train_loader:
        _, hits, labels = data
        optim.zero_grad()

        # Make prediction
        hits = hits.to(DEVICE)
        labels = labels.to(DEVICE)
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.flatten(pred[~padding_mask])
        labels = labels[~padding_mask]

        # Calculate loss and use it to update weights
        loss = loss_fn(pred, labels)
        loss.backward()
        optim.step()
        losses += loss.item()

    return losses / len(train_loader)

def evaluate(model, validation_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses = 0.
    with torch.no_grad():
        for data in validation_loader:
            _, hits, labels = data

            # Make prediction
            hits = hits.to(DEVICE)
            labels = labels.to(DEVICE)
            
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            pred = model(hits, padding_mask)

            pred = torch.unsqueeze(pred[~padding_mask], 0)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)
            
            loss = loss_fn(pred, labels)
            losses += loss.item()
            
    return losses / len(validation_loader)

def predict(model, cluster_hits):
    '''
    Evaluates the network on the test data. Returns the predictions
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()

    # Make prediction
    cluster_hits = cluster_hits.to(DEVICE)

    padding_mask = (cluster_hits == PAD_TOKEN).all(dim=2)
    pred = model(cluster_hits, padding_mask)

    cluster_hits = torch.unsqueeze(pred[~padding_mask], 0)
    pred = torch.unsqueeze(pred[~padding_mask], 0)
    refined_hits = cluster_hits[pred]

    return refined_hits

if __name__ == "__main__":
    NUM_EPOCHS = 5
    EARLY_STOPPING = 100
    MODEL_NAME = "refiner"

    torch.manual_seed(37)  # for reproducibility

    # Load dataset into dataloader
    hits_data, labels_data = load_calibration_data(data_path="predictions.csv")
    dataset = ClustersDataset(hits_data, labels_data)
    train_loader, valid_loader = get_dataloaders(dataset,
                                                train_frac=0.8,
                                                valid_frac=0.2,
                                                batch_size=64)
    print("data loaded")

    # Transformer model
    transformer = RefinerTransformer(num_encoder_layers=3,
                                        d_model=32,
                                        n_head=4,
                                        input_size=3,
                                        output_size=1,
                                        dim_feedforward=64,
                                        dropout=0.1)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn)

        # Evaluate using validation split
        val_loss = evaluate(transformer, valid_loader, loss_fn)

        # Bookkeeping
        print(f"Epoch: {epoch}\nVal loss: {val_loss:.8f}, Train loss: {train_loss:.8f}", flush=True)
        train_losses.append(train_loss)

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count, MODEL_NAME)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count, MODEL_NAME)
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break

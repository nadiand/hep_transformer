import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

from model import TransformerClassifier, PAD_TOKEN, save_model
from scoring import calc_score
from refining_clusters_dataloader import ClustersDataset, load_calibration_data, get_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clustering(pred_params):
    clustering_algorithm = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1)
    cluster_labels = []
    for _, event_prediction in enumerate(pred_params):
        regressed_params = np.array(event_prediction.tolist())
        event_cluster_labels = clustering_algorithm.fit_predict(regressed_params)
        cluster_labels.append(event_cluster_labels)

    cluster_labels = [torch.from_numpy(cl_lbl).int() for cl_lbl in cluster_labels]
    return cluster_labels

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
        _, hits, track_params, _ = data
        optim.zero_grad()

        # Make prediction
        hits = hits.to(DEVICE)
        track_params = track_params.to(DEVICE)
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)

        # Calculate loss and use it to update weights
        loss = loss_fn(pred, track_params)
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
            _, hits, track_params, _ = data

            # Make prediction
            hits = hits.to(DEVICE)
            track_params = track_params.to(DEVICE)
            
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            pred = model(hits, padding_mask)

            # pred = torch.unsqueeze(pred[~padding_mask], 0)
            # track_params = torch.unsqueeze(track_params[~padding_mask], 0)
            
            loss = loss_fn(pred, track_params)
            losses += loss.item()
            
    return losses / len(validation_loader)

def predict(model, test_loader):
    '''
    Evaluates the network on the test data. Returns the predictions
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score = 0.
    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Make prediction
        hits = hits.to(DEVICE)
        track_params = track_params.to(DEVICE)
        track_labels = track_labels.to(DEVICE)

        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        cluster_labels = clustering(pred)
        event_score = calc_score(cluster_labels, track_labels)
        score += event_score

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)
            to_store = []
            for i in range(len(hits[0])):
                to_store.append([hits[0][i][0].item(), hits[0][i][1].item(), hits[0][i][2].item(), cluster_labels[0][i].item(), track_labels[0][i][0].item(), event_id.item()])
            df = pd.DataFrame(to_store)
            df.to_csv('predictions.csv', mode='a', index=False, header=False)

    return predictions, score/len(test_loader)

if __name__ == "__main__":
    NUM_EPOCHS = 500
    EARLY_STOPPING = 100
    MODEL_NAME = "test"

    torch.manual_seed(37)  # for reproducibility

    # Load dataset into dataloader
    hits_data, track_params_data, track_classes_data = load_calibration_data(data_path="predictions.csv")
    dataset = ClustersDataset(hits_data, track_params_data, track_classes_data)
    train_loader, valid_loader = get_dataloaders(dataset,
                                                train_frac=0.8,
                                                valid_frac=0.2,
                                                batch_size=64)
    print("data loaded")

    # Transformer model
    transformer = TransformerClassifier(num_encoder_layers=6,
                                        d_model=32,
                                        n_head=4,
                                        input_size=3,
                                        output_size=3,
                                        dim_feedforward=128,
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

        # Bookkeeping
        print(f"Epoch: {epoch}, Train loss: {train_loss:.8f}", flush=True)
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

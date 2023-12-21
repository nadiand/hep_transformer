import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN
import pandas as pd
from torch.utils.data import DataLoader

from model_flashattn import TransformerClassifier, PAD_TOKEN, save_model
from dataset import HitsDataset, get_dataloaders, load_linear_2d_data, load_linear_3d_data, load_curved_3d_data
from scoring import calc_score, calc_score_trackml
from trackml_data import load_trackml_data
from sklearn.metrics import pairwise_distances

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clustering(pred_params):
    # def angdiff(point1, point2):
    #     return np.abs((((point1[1]-point2[1]) + np.pi) % (2*np.pi)) - np.pi)
    
    # def sim_affinity(X):
    #     return pairwise_distances(X, metric=angdiff)

    # clustering_algorithm = HDBSCAN()
    clustering_algorithm = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1) #, affinity='precomputed', linkage='single')
    cluster_labels = []
    for _, event_prediction in enumerate(pred_params):
        regressed_params = np.array(event_prediction.tolist())
        # X = sim_affinity(regressed_params)
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
        _, hits_orig, track_params, _ = data
        optim.zero_grad()

        # Make prediction
        hits = hits_orig.clone()
        hits = hits.to(DEVICE)
        track_params = track_params.to(DEVICE)
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)
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
        for data in valid_loader:
            _, hits_orig, track_params, _ = data

            # Make prediction
            hits = hits_orig.clone()
            hits = hits.to(DEVICE)
            track_params = track_params.to(DEVICE)
            
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            pred = model(hits, padding_mask)
            track_params = track_params[:, :pred.shape[1] ,:] 

            # Calculate loss and use it to update weights
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
        event_id, hits_orig, track_params, track_labels = data

        # Make prediction
        hits = hits_orig.clone()
        hits = hits.to(DEVICE)
        track_params = track_params.to(DEVICE)
        
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)
        track_params = track_params[:, :pred.shape[1] ,:]
        track_labels = track_labels[:, :pred.shape[1]]
        hits = hits[:, :pred.shape[1], :]

        cluster_labels = clustering(pred)
        event_score = calc_score_trackml(cluster_labels, track_labels)
        score += event_score

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)

    return predictions, score/len(test_loader)

if __name__ == "__main__":
    NUM_EPOCHS = 1
    EARLY_STOPPING = 50
    MODEL_NAME = "test"
    max_nr_hits = 800

    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets, and get dataloaders
    # hits_data, track_params_data, track_classes_data = load_trackml_data(data_path="nr_hits.csv", max_num_hits=max_nr_hits)
    # dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    # train_loader, valid_loader, test_loader = get_dataloaders(dataset,
    #                                                           train_frac=0.7,
    #                                                           valid_frac=0.15,
    #                                                           test_frac=0.15,
    #                                                           batch_size=2)
    print("data loaded")

    # Transformer model
    transformer = TransformerClassifier(num_encoder_layers=6,
                                        d_model=32,
                                        n_head=8,
                                        input_size=3,
                                        output_size=3,
                                        dim_feedforward=128,
                                        dropout=0.1)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_losses = []
        with pd.read_csv("trackml_train_data_subdivided.csv", chunksize=4) as reader:
            for chunk in reader:
                hits_data, track_params_data, track_classes_data = load_trackml_data(chunk)
                dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
                train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
                loss = train_epoch(transformer, optimizer, train_loader, loss_fn)
                train_losses.append(loss)
        train_loss = np.array(train_losses).mean()

        validation_losses = []
        with pd.read_csv("trackml_validation_data_subdivided.csv", chunksize=4) as reader:
            for chunk in reader:
                hits_data, track_params_data, track_classes_data = load_trackml_data(chunk)
                dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
                valid_loader = DataLoader(dataset, batch_size=8, shuffle=False)
                loss = evaluate(transformer, valid_loader, loss_fn)
                validation_losses.append(loss)
        val_loss = np.array(validation_losses).mean()
        
        print(f"Epoch: {epoch}\nVal loss: {val_loss:.8f}, Train loss: {train_loss:.8f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

    scores = []
    with pd.read_csv("trackml_test_data_subdivided.csv", chunksize=4) as reader:
        for chunk in reader:
            hits_data, track_params_data, track_classes_data = load_trackml_data(chunk)
            dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            preds, score = predict(transformer, test_loader)
            scores.append(score)
    print(np.array(scores).mean())

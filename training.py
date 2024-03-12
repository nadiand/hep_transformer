import torch
import torch.nn as nn
import numpy as np
from hdbscan import HDBSCAN
import pandas as pd

from model import TransformerClassifier, PAD_TOKEN, save_model
from dataset import HitsDataset, get_dataloaders, load_linear_2d_data, load_linear_3d_data, load_curved_3d_data
from scoring import calc_score, calc_score_trackml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clustering(pred_params, min_cl_size, min_samples):
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
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
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            pred = model(hits, padding_mask)

            pred = torch.unsqueeze(pred[~padding_mask], 0)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)
            
            loss = loss_fn(pred, track_params)
            losses += loss.item()
            
    return losses / len(validation_loader)

def predict(model, test_loader, min_cl_size, min_samples):
    '''
    Evaluates the network on the test data. Returns the predictions
    '''
    # Get the network in evaluation mode
    torch.set_grad_enabled(False)
    model.eval()
    predictions = {}
    score, perfects, doubles, lhcs = 0., 0., 0., 0.
    for data in test_loader:
        event_id, hits, track_params, track_labels = data

        # Make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        pred = model(hits, padding_mask)

        hits = torch.unsqueeze(hits[~padding_mask], 0)
        pred = torch.unsqueeze(pred[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        # noise = np.random.uniform(-0.04, 0.04, size=(track_params.shape[0], track_params.shape[1], track_params.shape[2]))
        # track_params += noise

        cluster_labels = clustering(pred, min_cl_size, min_samples)
        event_score, scores = calc_score_trackml(cluster_labels[0], track_labels[0])
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)
            # to_store = []
            # for i in range(len(hits[0])):
            #     to_store.append([hits[0][i][0].item(), hits[0][i][1].item(), hits[0][i][2].item(), cluster_labels[0][i].item(), track_labels[0][i][0].item(), event_id.item()])
            # df = pd.DataFrame(to_store)
            # df.to_csv('predictions.csv', mode='a', index=False, header=False)

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)

if __name__ == "__main__":
    NUM_EPOCHS = 5
    EARLY_STOPPING = 100
    MODEL_NAME = "test"
    MAX_NUM_HITS = 100

    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets, and get dataloaders
    hits_data, track_params_data, track_classes_data = load_curved_3d_data(data_path="hits_and_tracks_3d_3curved_events_all.csv", max_num_hits=MAX_NUM_HITS)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
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

    loss_fn = nn.MSELoss()
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

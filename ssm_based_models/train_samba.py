import torch
import torch.nn as nn
import numpy as np
import argparse
from hdbscan import HDBSCAN

from samba_model import Model
from load_sim_data import HitsDataset, get_dataloaders, load_trackml_data, PAD_TOKEN
from evaluation.scoring import calc_score, calc_score_trackml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clustering(pred_params, min_cl_size, min_samples):
    '''
    Function to perform HDBSCAN on the predicted track parameters, with specified
    HDBSCAN hyperparameters. Returns the associated cluster IDs.
    '''
    clustering_algorithm = HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples)
    cluster_labels = []
    for _, event_prediction in enumerate(pred_params):
        regressed_params = np.array(event_prediction.tolist())
        event_cluster_labels = clustering_algorithm.fit_predict(regressed_params)
        cluster_labels.append(event_cluster_labels)

    cluster_labels = [torch.from_numpy(cl_lbl).int() for cl_lbl in cluster_labels]
    return cluster_labels

def train_epoch(model, optim, train_loader, loss_fn, scaler):
    '''
    Conducts a single epoch of training: prediction, loss calculation, and loss
    backpropagation. Returns the average loss over the whole train data.
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    intermid_loss = 0.
    optim.zero_grad()

    for i, data in enumerate(train_loader):
        _, hits, track_params, _ = data
        # Make prediction
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        hits = torch.unsqueeze(hits[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        with torch.cuda.amp.autocast():
            pred = model(hits, padding_mask)
            loss = loss_fn(pred, track_params)

        # Calculate loss and use it to update weights
        intermid_loss += loss
        if (i+1) % 16 == 0:
            mean_loss = intermid_loss.mean()
            scaler.scale(mean_loss).backward()
            scaler.step(optim)
            scaler.update()
            losses += mean_loss.item()
            intermid_loss = 0.
            optim.zero_grad()

    return losses / len(train_loader)


def evaluate(model, validation_loader, loss_fn):
    '''
    Evaluates the network on the validation data by making a prediction and
    calculating the loss. Returns the average loss over the whole val data.
    '''
    # Get the network in evaluation mode
    model.eval()
    losses = 0.
    intermid_loss = 0.
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            _, hits, track_params, _ = data

            # Make prediction
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            hits = torch.unsqueeze(hits[~padding_mask], 0)
            track_params = torch.unsqueeze(track_params[~padding_mask], 0)
            with torch.cuda.amp.autocast():
                pred = model(hits, padding_mask)
                loss = loss_fn(pred, track_params)

            # Update loss after a "batch"
            intermid_loss += loss
            if (i+1) % 16 == 0:
                mean_loss = intermid_loss.mean()
                losses += mean_loss.item()
                intermid_loss = 0.

    return losses / len(validation_loader)


def predict(model, test_loader, min_cl_size, min_samples):
    '''
    Evaluates the network on the test data. Returns the predictions and scores.
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
        hits = torch.unsqueeze(hits[~padding_mask], 0)
        track_params = torch.unsqueeze(track_params[~padding_mask], 0)
        track_labels = torch.unsqueeze(track_labels[~padding_mask], 0)

        with torch.cuda.amp.autocast():
            pred = model(hits, padding_mask)

        cluster_labels = clustering(pred, min_cl_size, min_samples)
        event_score, scores = calc_score_trackml(cluster_labels[0], track_labels[0])
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        for _, e_id in enumerate(event_id):
            predictions[e_id.item()] = (hits, pred, track_params, cluster_labels, track_labels, event_score)

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_epochs', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--nr_enc_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--nr_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(37)  # for reproducibility

    # Load and split dataset into training, validation and test sets, and get dataloaders
    hits_data, track_params_data, track_classes_data = load_trackml_data(data=args.data_path)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
    print("Data loaded", flush=True)
    print(len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset), flush=True)

    # Samba model
    transformer = Model(n_layers=args.nr_enc_layers,
                                        d_model=args.embedding_size,
                                        n_head=args.nr_heads,
                                        input_size=3,
                                        output_size=4,
                                        layer_norm_eps=0.1,
                                        local_window=500,
                                        full_per_layer=2)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params), flush=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(args.nr_epochs):
        print(f"epoch {epoch}", flush=True)
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn, scaler)
        print(f"train epoch over", flush=True)
        # Evaluate using validation split
        val_loss = evaluate(transformer, valid_loader, loss_fn)
        print(f"eval over", flush=True)

        # Bookkeeping
        print(f"Epoch: {epoch}\nVal loss: {val_loss:.8f}, Train loss: {train_loss:.8f}", flush=True)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            # save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count, args.model_name)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            # save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count, args.model_name)
            count += 1

        if count >= args.early_stop:
            print("Early stopping...")
            break
import torch
import torch.nn as nn
import numpy as np
from hdbscan import HDBSCAN
import argparse
from time import process_time_ns, perf_counter_ns

from model import TransformerRegressor, save_model
from ssm_based_models.load_sim_data import HitsDataset, get_dataloaders, PAD_TOKEN, load_trackml_data
from evaluation.scoring import calc_score_trackml
from custom_encoder import generate_flex_padding_mask

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
    scaler is necessary to ensure the model's convergence (necessary due to usage
    of mixed precision, needed for Flash attention).
    '''
    # Get the network in train mode
    torch.set_grad_enabled(True)
    model.train()
    losses = 0.
    optim.zero_grad()

    for i, data in enumerate(train_loader):
        _, hits, seqlens, track_params, _ = data

        # Make masks
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        flex_padding_mask = generate_flex_padding_mask(seqlens)

        # Make prediction
        with torch.amp.autocast('cuda'):
            pred = model(hits, padding_mask, f'train_{i}', flex_padding_mask)

            # Unpad for loss calculation
            batched_pred = []
            batched_target = []
            B = len(seqlens)

            for b_idx in range(B):
                if B == 1:
                    seq_len = seqlens.item()
                else:
                    seq_len = seqlens[b_idx].item()
                # unpad just [0..seq_len) for pred
                this_pred = pred[b_idx, :seq_len, :]
                this_target = track_params[b_idx, :seq_len, :]
                batched_pred.append(this_pred)
                batched_target.append(this_target)

            final_pred = torch.cat(batched_pred, dim=0)
            targets = torch.cat(batched_target, dim=0)

            loss = loss_fn(final_pred, targets)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        losses += loss.item()
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
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            _, hits, seqlens, track_params, _ = data

            # Make masks
            padding_mask = (hits == PAD_TOKEN).all(dim=2)
            flex_padding_mask = generate_flex_padding_mask(seqlens)

            with torch.amp.autocast('cuda'):
                pred = model(hits, padding_mask, f'valid_{i}', flex_padding_mask)

                # Unpad for loss calculation
                batched_pred = []
                batched_target = []
                B = len(seqlens)

                for b_idx in range(B):
                    if B == 1:
                        seq_len = seqlens.item()
                    else:
                        seq_len = seqlens[b_idx].item()
                    # unpad just [0..seq_len) for pred
                    this_pred = pred[b_idx, :seq_len, :]
                    this_target = track_params[b_idx, :seq_len, :]
                    batched_pred.append(this_pred)
                    batched_target.append(this_target)

                final_pred = torch.cat(batched_pred, dim=0)
                targets = torch.cat(batched_target, dim=0)

                loss = loss_fn(final_pred, targets)

            losses += loss.item()

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

    # Time performance bookkeeping
    cuda_times = []
    cpu_times = []
    cpu_prep_times = []
    total_times = []
    mask_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_mask = torch.cuda.Event(enable_timing=True)
    end_mask = torch.cuda.Event(enable_timing=True)

    for i, data in enumerate(test_loader):
        total_start = perf_counter_ns()
        event_id, hits, seqlens, track_params, track_labels = data
        start_event.record()

        # Make masks
        padding_mask = (hits == PAD_TOKEN).all(dim=2)
        start_mask.record()
        flex_padding_mask = generate_flex_padding_mask(seqlens)
        end_mask.record()
        torch.cuda.synchronize()
        mask_elapsed = start_mask.elapsed_time(end_mask)
        mask_times.append(mask_elapsed)

        with torch.amp.autocast('cuda'):
            pred = model(hits, padding_mask, f'test_{i}', flex_padding_mask)
            end_event.record()
            torch.cuda.synchronize()
            cuda_elapsed = start_event.elapsed_time(end_event)  # in ms
            cuda_times.append(cuda_elapsed)

            prep_start = process_time_ns()
            # Unpad for score calculation and plotting
            batched_hits = []
            batched_pred = []
            batched_target = []
            batched_classes = []
            B = len(seqlens)

            for b_idx in range(B):
                if B == 1:
                    seq_len = seqlens.item()
                else:
                    seq_len = seqlens[b_idx].item()
                # unpad just [0..seq_len) for pred
                this_hits = hits[b_idx, :seq_len, :]
                this_pred = pred[b_idx, :seq_len, :]
                this_target = track_params[b_idx, :seq_len, :]
                this_labels = track_labels[b_idx, :seq_len, :]
                batched_hits.append(this_hits)
                batched_pred.append(this_pred)
                batched_target.append(this_target)
                batched_classes.append(this_labels)

            hits = torch.cat(batched_hits, dim=0)
            final_pred = torch.cat(batched_pred, dim=0)
            targets = torch.cat(batched_target, dim=0)
            classes = torch.cat(batched_classes, dim=0)
            hits = torch.unsqueeze(hits, 0)
            final_pred = torch.unsqueeze(final_pred, 0)
            targets = torch.unsqueeze(targets, 0)
            classes = torch.unsqueeze(classes, 0)

        prep_end = process_time_ns()
        cpu_prep_times.append(prep_end - prep_start)

        # Cluster and evaluate
        start_cpu_time = process_time_ns()
        cluster_labels = clustering(final_pred, min_cl_size, min_samples)
        end_cpu_time = process_time_ns()
        cpu_times.append(end_cpu_time - start_cpu_time)

        event_score, scores = calc_score_trackml(cluster_labels[0], classes[0])
        score += event_score
        perfects += scores[0]
        doubles += scores[1]
        lhcs += scores[2]

        predictions[event_id.item()] = (hits, final_pred, targets, cluster_labels, classes, event_score)

        total_end = perf_counter_ns()
        total_times.append(total_end - total_start)

    print("Avg CUDA forward time (ms):", sum(cuda_times[1:]) / len(cuda_times[1:]))
    print("Avg CUDA masking time (ms):", sum(mask_times[1:]) / len(mask_times[1:]))
    print("Avg CPU prep time (ns):", sum(cpu_prep_times[1:]) / len(cpu_prep_times[1:]))
    print("Avg CPU clustering time (ns):", sum(cpu_times[1:]) / len(cpu_times[1:]))
    print("Avg total latency (ns):", sum(total_times[1:]) / len(total_times[1:]))
    print()

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_epochs', type=int, default=50)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--max_nr_hits', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--nr_enc_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--nr_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(37)  # for reproducibility

    # Loading data
    hits_data, seqlen_data, track_params_data, track_classes_data = load_trackml_data(data=args.data_path)
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data, seqlen_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=16)
    print("Data loaded")

    # Transformer model
    transformer = TransformerRegressor(num_encoder_layers=args.nr_enc_layers,
                                        d_model=args.embedding_size,
                                        n_head=args.nr_heads,
                                        input_size=3,
                                        output_size=4,
                                        dim_feedforward=args.hidden_dim,
                                        dropout=args.dropout,
                                        use_flashattn=True)
    transformer = transformer.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(pytorch_total_params))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    # Training
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    count = 0

    for epoch in range(args.nr_epochs):
        # Train the model
        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn, scaler)

        # Evaluate using validation split
        val_loss = evaluate(transformer, valid_loader, loss_fn)

        print(f"Epoch: {epoch}\nVal loss: {val_loss:.8f}, Train loss: {train_loss:.8f}", flush=True)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            # If the model has a new best validation loss, save it as "the best"
            min_val_loss = val_loss
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count, args.model_name)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count, args.model_name)
            count += 1

        if count >= args.early_stopping:
            print("Early stopping...")
            break

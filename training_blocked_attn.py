import torch
import torch.nn as nn
import numpy as np
from hdbscan import HDBSCAN
from time import process_time_ns, perf_counter_ns

from model import TransformerRegressor, save_model
from scoring import calc_score_trackml
from blocked_load_data import HitsDataset, get_dataloaders, PAD_TOKEN, load_trackml_data, build_fixed_class_mask, build_full_mask_fn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fixed_blocked_class_mask = build_fixed_class_mask()
fixed_blocked_class_mask = fixed_blocked_class_mask.to(DEVICE)

def clustering(pred_params, min_cl_size, min_samples):
    """
    Function to perform HDBSCAN on the predicted track parameters, with specified
    HDBSCAN hyperparameters. Returns the associated cluster IDs.
    """
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
        _, hits, class_info, track_params, _ = data

        # Make mask
        flex_mask = build_full_mask_fn(fixed_blocked_class_mask, class_info)

        # Make prediction
        with torch.amp.autocast('cuda'):
            pred = model(hits, padding_mask=None, batch_name=f'train_{i}', flex_padding_mask=flex_mask)
            final_preds = pred[class_info]
            targets = track_params[class_info]

            loss = loss_fn(final_preds, targets)
            print("class_info true count:", class_info.sum().item(),
                  "total:", class_info.numel(),
                  "fraction valid:", class_info.sum().item() / class_info.numel())
            print("pred shape:", pred.shape, "final_preds shape:", final_preds.shape)
            print("-"*80)

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
            _, hits, class_info, track_params, _ = data

            # Make mask
            flex_mask = build_full_mask_fn(fixed_blocked_class_mask, class_info)

            with torch.amp.autocast('cuda'):
                pred = model(hits, padding_mask=None, batch_name=f'valid_{i}', flex_padding_mask=flex_mask)

                # Unpad for loss calculation
                final_preds = pred[class_info]
                targets = track_params[class_info]
                loss = loss_fn(final_preds, targets)

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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i, data in enumerate(test_loader):
        total_start = perf_counter_ns()
        event_id, hits, class_info, track_params, track_labels = data
        start_event.record()

        # Make mask
        flex_mask = build_full_mask_fn(fixed_blocked_class_mask, class_info)

        with torch.amp.autocast('cuda'):
            print("hits", hits.shape)
            pred = model(hits, padding_mask=None, batch_name=f'test_{i}', flex_padding_mask=flex_mask)
            print("pred", pred.shape)
            end_event.record()
            torch.cuda.synchronize()
            cuda_elapsed = start_event.elapsed_time(end_event)  # in ms
            cuda_times.append(cuda_elapsed)

            prep_start = process_time_ns()
            # Unpad for score calculation and plotting
            batched_pred = pred[class_info]
            batched_target = track_params[class_info]
            batched_classes = track_labels[class_info]
            batched_hits = hits[class_info]
            print("after class info hits", batched_hits.shape)
            print("after class info pred", batched_pred.shape)

            hits = torch.unsqueeze(batched_hits, 0)
            final_pred = torch.unsqueeze(batched_pred, 0)
            targets = torch.unsqueeze(batched_target, 0)
            classes = torch.unsqueeze(batched_classes, 0)
            print("final pred", final_pred.shape)
            if final_pred.shape[1] == 0:
                print(class_info)
                print(torch.sum(class_info))
            print("-"*80)

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
    print("Avg CPU prep time (ns):", sum(cpu_prep_times[1:]) / len(cpu_prep_times[1:]))
    print("Avg CPU clustering time (ns):", sum(cpu_times[1:]) / len(cpu_times[1:]))
    print("Avg total latency (ns):", sum(total_times[1:]) / len(total_times[1:]))

    return predictions, score/len(test_loader), perfects/len(test_loader), doubles/len(test_loader), lhcs/len(test_loader)


if __name__ == "__main__":
    NUM_EPOCHS = 50 #500
    EARLY_STOPPING = 10
    MODEL_NAME = "test" #"flex_encreg_7layer_128dmodel_4head_256dimff_01drop_newdata_batch16_padding0shuffletrued_spherical"
    MAX_NUM_HITS = 6000 #5000

    torch.manual_seed(37)  # for reproducibility

    # Loading data
    hits_data, seqlen_data, track_params_data, track_classes_data = load_trackml_data(data="/projects/0/nisei0750/nadia/trackML_200_500_40k_events.csv")
    dataset = HitsDataset(hits_data, track_params_data, track_classes_data, seqlen_data)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=16)

    # Transformer model
    transformer = TransformerRegressor(num_encoder_layers=7,
                                        d_model=128,
                                        n_head=4,
                                        input_size=3,
                                        output_size=4,
                                        dim_feedforward=256,
                                        dropout=0.1,
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

    for epoch in range(NUM_EPOCHS):
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
            save_model(transformer, optimizer, "best", val_losses, train_losses, epoch, count, MODEL_NAME)
            count = 0
        else:
            # If the model's validation loss isn't better than the best, save it as "the last"
            save_model(transformer, optimizer, "last", val_losses, train_losses, epoch, count, MODEL_NAME)
            count += 1

        if count >= EARLY_STOPPING:
            print("Early stopping...")
            break

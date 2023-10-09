import torch
from model import TransformerClassifier
from training import predict
from dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from plotting import *
from training import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hits_data, track_params_data, track_classes_data = load_linear_3d_data(data_path="hits_and_tracks_3d_events_all.csv", max_num_hits=81) #540
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1024)
print('data loaded')

number_encoder_layers = [2, 3, 4, 5, 6, 7, 8]
embedding_dimensions = [8, 16, 32, 64, 128]
number_attention_heads = [4, 8, 16, 32]
dim_feedforward_layes = [64, 128, 256]
dropout_percentage = [0., 0.1, 0.2]

for num_enc_l in number_encoder_layers:
    for emb_dim in embedding_dimensions:
        for n_head in number_attention_heads:
            for dim_ff in dim_feedforward_layes:
                for drop in dropout_percentage:
                    if emb_dim % n_head != 0:
                        continue

                    print(f"Hyperparam set: {num_enc_l} encoder layers, {emb_dim} d_model, {n_head} heads, {dim_ff} dim of feedforward, {drop}% dropout")
                    transformer = TransformerClassifier(num_encoder_layers=num_enc_l,
                                                        d_model=emb_dim,
                                                        n_head=n_head,
                                                        input_size=3,
                                                        output_size=2,
                                                        dim_feedforward=dim_ff,
                                                        dropout=drop)
                    transformer = transformer.to(DEVICE)
                    pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
                    print("Total trainable params: {}".format(pytorch_total_params))

                    loss_fn = nn.MSELoss()
                    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

                    # Training
                    train_losses, val_losses = [], []
                    min_val_loss = np.inf
                    count = 0
                    NUM_EPOCHS = 100
                    EARLY_STOPPING = 20

                    for epoch in range(NUM_EPOCHS):
                        # Train the model
                        train_loss = train_epoch(transformer, optimizer, train_loader, loss_fn, True)
                        val_loss = evaluate(transformer, valid_loader, loss_fn, True)

                        train_losses.append(train_loss)
                        val_losses.append(val_loss)

                        if val_loss < min_val_loss:
                            # If the model has a new best validation loss, save it as "the best"
                            min_val_loss = val_loss
                            count = 0
                        else:
                            # If the model's validation loss isn't better than the best, save it as "the last"
                            count += 1

                        if count >= EARLY_STOPPING:
                            print(f"Early stopping in epoch {epoch}\n")
                            break

                    preds, score = predict(transformer, test_loader, True)
                    print(f"Model has score {score}\n")

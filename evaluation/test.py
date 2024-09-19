import torch

from model import TransformerRegressor
from training import predict
from data_processing.dataset import HitsDataset, get_dataloaders, load_linear_3d_data, load_linear_2d_data, load_curved_3d_data
from data_processing.trackml_data import load_trackml_data, load_trackml_data_pt
from evaluation.plotting import plot_heatmap


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = TransformerRegressor(num_encoder_layers=6,
                                    d_model=32,
                                    n_head=4,
                                    input_size=3,
                                    output_size=4,
                                    dim_feedforward=128,
                                    dropout=0.1) #TODO make possible to pass the use_flashattm arg !!
transformer = transformer.to(DEVICE)

checkpoint = torch.load("best_models/10to50_tml_best", map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

hits_data, track_params_data, track_classes_data = load_trackml_data(data="trackml_10to50tracks_40kevents.csv", max_num_hits=700, normalize=True)
dataset = HitsDataset(hits_data, track_params_data, track_classes_data)
train_loader, valid_loader, test_loader = get_dataloaders(dataset,
                                                              train_frac=0.7,
                                                              valid_frac=0.15,
                                                              test_frac=0.15,
                                                              batch_size=1)
print('data loaded')

for cl_size in [2, 3, 4, 5, 6, 7, 8, 9]:
    for min_sam in [2, 3, 4, 5, 6, 7, 8, 9]:
        preds, score, perfect, double_maj, lhc = predict(transformer, test_loader, cl_size, min_sam)
        print(f'cluster size {cl_size}, min samples {min_sam}, score {score}', flush=True)
        print(perfect, double_maj, lhc, flush=True)

        if cl_size == 5 and min_sam == 2:
            preds = list(preds.values())
            for param in ["theta", "sinphi", "cosphi", "q"]:
                plot_heatmap(preds, param, "10to50_tml") # TODO fix make the file name passable! 
                # TODO just convert all files to be used with arguments from commandline
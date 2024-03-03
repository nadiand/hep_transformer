import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

colors = {-1: 'yellow', 0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'brown', 5: 'orange', 6: 'beige', 7: 'magenta', 8: 'gray', 9: 'pink', 10: 'indigo', 11: 'maroon', 12: 'coral', 13: 'lime', 14: 'deepskyblue', 15: 'gold', 16: 'lightgray', 17: 'plum', 18: 'tan', 19: 'yellowgreen', 20: 'darkgreen', 21: 'darkblue', 22: 'lightgreen', 23: 'lightblue', 24: 'fuchsia', 25: 'wheat', 26: 'lawngreen', 27: 'steelblue', 28: 'orchid', 29: 'slategray', 30: 'peru', 31: 'tomato', 32: 'seagreen', 33: 'mediumblue', 34: 'silver', 35: 'firebrick', 36: 'springgreen', 37: 'darkolivegreen', 38: 'dodgerblue', 39: 'crimson', 40: 'indigo', 41: 'orangered', 42: 'aquamarine', 43: 'lemonchiffon', 44: 'gainsboro', 45: 'darkcyan', 46: 'darkmagenta', 47: 'lightpink', 48: 'navajowhite', 49: 'cornflowerblue', 50: 'mistyrose'}

def plot_score_vs_tracknr(preds):
    scores = [0]*50
    for event in preds:
        track_nr = max(event[4][0])
        score = event[5]
        so_far = scores[track_nr]
        if so_far > 0:
            scores[track_nr] = (so_far+score)/2
        else:
            scores[track_nr] = score

    plt.figure()
    plt.bar(np.arange(0, 50), scores)
    plt.xlabel("Nr of track")
    plt.ylabel("Score")
    plt.show()


def visualize_track(coords, labels, label, nr):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Group the hits originating from the same track together
    tracks = {}
    for i, coord in enumerate(coords):
        if coord[0] != -1.:
            x, y, z = convert_cylindrical_to_cartesian(coord[0], coord[1], coord[2])
            track_id = labels[i].item() % 50
            if not track_id in tracks.keys():
                tracks[track_id] = [(x.item(), y.item(), z.item())]
            else:
                values = tracks[track_id]
                if not (x.item(), y.item(), z.item()) in values:
                    values.append((x.item(), y.item(), z.item()))
                    tracks.update({track_id: values})

    i = 0
    for t in tracks.keys():
        if i < 10:
            i += 1
            # Take all hits associated with a track
            coords = tracks[t]
            # ref = (0,0,0)
            # coords.sort(key=lambda x: (x[0] - ref[0]) ** 2 + (x[1] - ref[1]) ** 2 + (x[2] - ref[2]) ** 2)

            xs = [coord[0] for coord in coords]
            ys = [coord[1] for coord in coords]
            zs = [coord[2] for coord in coords]

            unique_indices = np.unique(zs, return_index=True)[1]
            xs = [xs[i] for i in unique_indices]
            ys = [ys[i] for i in unique_indices]
            zs = [zs[i] for i in unique_indices]

            # Normalize the coordinates
            # xs = (xs - np.mean(xs))/np.std(xs)
            # ys = (ys - np.mean(ys))/np.std(ys)
            # zs = (zs - np.mean(zs))/np.std(zs)

            interp_func_x = interp1d(zs, xs, kind='cubic')
            interp_func_y = interp1d(zs, ys, kind='cubic')
            interp_func_z = interp1d(zs, zs, kind='cubic')

            # Define the range for the interpolated curve
            # print(zs.min(), zs.max())
            interp_z = np.linspace(min(zs), max(zs), 500)

            # Interpolate coordinates
            interp_x = interp_func_x(interp_z)
            interp_y = interp_func_y(interp_z)
            interp_z = interp_func_z(interp_z)

            # Plot
            ax.plot(interp_x, interp_y, interp_z, linestyle='-', color=colors[t])
            ax.scatter(interp_x, interp_y, interp_z, marker=".", color=colors[t])
    
    # plt.savefig(f"figs/{label}{nr}.png")
    ax.set_xlim([-55000, 50000])
    ax.set_ylim([0, 55000])
    ax.set_zlim([0, 3000])
    plt.show()


def visualize_3d_hits(data_df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot events
    for ind in range(1): # 1 event in this case
        row = data_df[ind]
        print(row[0])
        coords, labels = row[1], row[3]
        for i, coord in enumerate(coords):
            if coord[0] != -1.:
                plt.plot(coord[0]*np.cos(coord[1]), coord[0]*np.sin(coord[1]), coord[2], marker=".", c=colors[labels[i].item()])
    
    plt.show()

def plot_3d_parameters(parameters, param):
    plt.figure()
    param_names = {'theta': 0, 'phi' : 1, 'pitch coeff': 1, 'radial coeff': 0, 'test':0}
    for event_parameters in parameters:
        predicted, true = event_parameters[1][0][:, param_names[param]], event_parameters[2][0][:, param_names[param]]
        indices = np.fabs(true) < 1
        predicted = predicted[indices]
        true = true[indices]
        plt.scatter(predicted, true, c='blue', alpha=0.2)

    plt.title(f"Regressed {param} vs ground truth")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_regressed_params(parameters):
    i = 0
    for event_parameters in parameters:
        plt.figure()
        predicted, true = event_parameters[1][0], event_parameters[2][0]
        for j in range(len(predicted)):
            plt.scatter(predicted[j, 0], predicted[j, 1], c=colors[event_parameters[3][0][j].item()], alpha=0.5, s=80)
            plt.scatter(true[j, 0], true[j, 1], edgecolors='black', marker='v', c=colors[event_parameters[3][0][j].item()], alpha=0.5, s=80)
        plt.title(f"Predicted parameters vs ground truth")
        plt.xlabel("radial coeff")
        plt.ylabel("pitch coeff")
        i += 1
        plt.show()


def plot_2d(parameters):
    for i in range(len(parameters)):
        plt.scatter(parameters[i][1][0], parameters[i][2][0])
    plt.title("Regressed parameters vs ground truth")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_cluster_labels_2d(parameters):
    for i in range(len(parameters)):
        print(parameters[i][3][0], parameters[i][4][0])
        plt.scatter(parameters[i][3][0], parameters[i][3][0], c='red')
        plt.scatter(parameters[i][4][0], parameters[i][4][0], c='blue', marker='v')
        plt.show()
    plt.title("Regressed parameters vs ground truth")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('plot_slope.png')

def convert_cylindrical_to_cartesian(r, theta, z):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x, y, z

def plot_results(hits, pred_cluster, true_cluster):
    x_vec, y_vec, z_vec = convert_cylindrical_to_cartesian(hits[:, 0], hits[:, 1], hits[:, 2])
    # x_vec_true, y_vec_true, z_vec_true = convert_cylindrical_to_cartesian(true_cluster[:, 0], true_cluster[:, 1], true_cluster[:, 2])
    # x_vec_pred, y_vec_pred, z_vec_pred = convert_cylindrical_to_cartesian(pred_cluster[:, 0], pred_cluster[:, 1], pred_cluster[:, 2])

    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')

    # Data for three-dimensional scattered points
    ax.scatter3D(x_vec, y_vec, z_vec, cmap='viridis', alpha=0.4)

    ax.plot3D(x_vec, y_vec, z_vec, 'green')

    # Data for three-dimensional scattered points
    # ax.scatter3D(x_vec_true[1:10], y_vec_true[1:10], z_vec_true[1:10], cmap='viridis')

    # ax.plot3D(x_vec_pred[1:10], y_vec_pred[1:10], z_vec_pred[1:10], 'red')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def plot_heatmap(parameters, param, name):
    plt.figure()
    param_names = {"theta":0, "phi":1, "q":2}
    all_pred, all_true = [], []
    for event_parameters in parameters:
        predicted, true = event_parameters[1][0][:, param_names[param]], event_parameters[2][0][:, param_names[param]]
        all_pred.append(predicted.cpu().tolist())
        all_true.append(true.cpu().tolist())
    all_pred_flattened = [item for sublist in all_pred for item in sublist]
    all_true_flattened = [item for sublist in all_true for item in sublist]

    heatmap, xedges, yedges = np.histogram2d(all_pred_flattened, all_true_flattened, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    transposed_heatmap = heatmap.T
    # A hack to make the points with no data white - setting the 0s to nans
    transposed_heatmap[transposed_heatmap == 0.0] = np.nan

    plt.clf()
    plt.imshow(transposed_heatmap, extent=extent, origin='lower')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Regressed {param} vs ground truth")
    plt.savefig(f"{param}_heatmap_{name}.png")
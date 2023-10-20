import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

colors = {-1: 'yellow', 0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'brown', 5: 'orange', 6: 'beige', 7: 'magenta', 8: 'gray', 9: 'pink', 10: 'indigo', 11: 'maroon', 12: 'coral', 13: 'lime', 14: 'deepskyblue', 15: 'gold', 16: 'lightgray', 17: 'plum', 18: 'tan', 19: 'yellowgreen', 20: 'darkgreen', 21: 'darkblue', 22: 'lightgreen', 23: 'lightblue'}

def visualize_3d_hits(data_df):
    '''
    Visualizes the simplified setup: the 5 detectors as spheres and a few 
    events as the hits of the generated particles with the detectors.
    '''
    # Plot the detectors
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
    param_names = {'theta': 0, 'phi' : 1, 'pitch coeff': 1, 'radial coeff': 0}
    for event_parameters in parameters:
        predicted, true = event_parameters[1][0], event_parameters[2][0]
        plt.scatter(predicted[:, param_names[param]], true[:, param_names[param]])

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
        plt.xlabel("theta")
        plt.ylabel("phi")
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
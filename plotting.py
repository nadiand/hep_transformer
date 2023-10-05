import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_3d(parameters, param):
    plt.figure()
    param_names = {'theta': 0, 'phi' : 1}
    for event_parameters in parameters:
        predicted, true = event_parameters[1][0], event_parameters[2][0]
        plt.scatter(predicted[:, param_names[param]], true[:, param_names[param]])

    plt.title(f"Regressed {param} vs ground truth")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_regressed_params(parameters, events):
    i = 0
    for event_parameters in parameters:
        plt.figure()
        predicted, true = event_parameters[1][0], event_parameters[2][0]
        colors = {-1: 'yellow', 0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'brown', 5: 'orange', 6: 'beige', 7: 'magenta', 8: 'gray', 9: 'pink', 10: 'indigo', 11: 'maroon', 12: 'coral', 13: 'lime', 14: 'deepskyblue', 15: 'gold', 16: 'lightgray', 17: 'plum', 18: 'tan', 19: 'yellowgreen'}
        for j in range(len(predicted)):
            plt.scatter(predicted[j, 0], predicted[j, 1], c=colors[event_parameters[3][0][j].item()])

        plt.scatter(true[:, 0], true[:, 1], c='black', marker='v')
        plt.title(f"Predicted parameters vs ground truth") #list(events)[i]
        plt.xlabel("theta")
        plt.ylabel("phi")
        i += 1
        plt.show()

def plot_cluster_labels(parameters, events):
    plt.figure()
    i = 0
    for event_parameters in parameters:
        predicted, true = event_parameters[3][0], event_parameters[4][0]
        plt.scatter(predicted, predicted, c='red')
        plt.scatter(true, true, c='blue', marker='v')
        plt.title(f"event {list(events)[i]}")
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
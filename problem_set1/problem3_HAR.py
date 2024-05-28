import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.metrics as skm
import pandas as pd
from math import exp

def generate_conf_matrix(predictions, labels, title): 
    confusion_matrix = skm.confusion_matrix(predictions, labels) 
    display_labels = [str(label) for label in np.unique(labels)]
    display = skm.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
    plt.figure()
    display.plot()
    plt.title(title)
    plt.show()
    return confusion_matrix

def data_visualization(predictions, labels, x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    marker_shapes = ['o', '^', 's', '.', 'x', '+', 'v', 'd', 'p', 'h', '8', '1', '2', '3', '4', 'H', 'D', '|', '_', '<', '>']
    for label, marker_shape in zip(np.unique(labels), marker_shapes):
        mask = labels == label
        ax.scatter(x[mask & (predictions != labels)], y[mask & (predictions != labels)], z[mask & (predictions != labels)], marker=marker_shape, color='red', label='Incorrect', s=10) 
        ax.scatter(x[mask & (predictions == labels)], y[mask & (predictions == labels)], z[mask & (predictions == labels)], marker=marker_shape, color='green', label='Correct', s=10)   
    ax.legend()
    plt.show()

if __name__ == '__main__':
    imported_data = pd.read_csv('X_train.txt', delim_whitespace=True, header=None).values
    imported_labels = pd.read_csv('y_train.txt', delim_whitespace=True, header=None).values
    class_priors =[np.sum(imported_labels == l) / imported_data.shape[0] for l in np.unique(imported_labels)]
    mean = []
    for label in np.unique(imported_labels):
        label_value = imported_data[imported_labels.flatten() == label] 
        mean_value = np.mean(label_value, axis=0) 
        mean.append(mean_value)
    covariance = [imported_data[imported_labels.flatten() == label] for label in np.unique(imported_labels)] 
    regularization_term = 0.1
    covariance = np.array([np.cov(i.T) + regularization_term * np.eye(i.T.shape[0]) for i in covariance])
    loss_matrix = np.ones((np.unique(imported_labels).shape[0], np.unique(imported_labels).shape[0])) - np.eye(np.unique(imported_labels).shape[0])
    predictions = np.argmin(loss_matrix.dot(np.diag(class_priors).dot(np.array([multivariate_normal.pdf(imported_data, np.array(mean)[i], covariance[i]) for i in range(len(np.array(mean)))]))), axis=0) + 1
    confusion_matrix = generate_conf_matrix(predictions, imported_labels, 'HAR Confusion Matrix')
    data_visualization(predictions, imported_labels.flatten(), imported_data[:, 0], imported_data[:, 1], imported_data[:, 2], 'HAR Data Visualization')
    error_probability = 1 - np.sum(np.diag(confusion_matrix)) / imported_data.shape[0]
    print(f'Error Probability: {error_probability:.3f}')
    

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.metrics as skm

def generate_samples(mean, covariance, class_priors, number_of_samples, component_mean, component_covariance):
    dim = mean.shape[1]
    generated_samples = np.zeros((number_of_samples, dim))
    generated_labels = np.random.choice(len(class_priors), size=number_of_samples, p=class_priors)
    for k in range(len(class_priors) - 1):
        indices = np.where(generated_labels == k)[0]
        generated_samples[indices] = np.random.multivariate_normal(mean[k], covariance[k], size=len(indices))
    indices = np.where(generated_labels == (len(class_priors) - 1))[0]
    mixture_component_labels = np.random.choice((len(class_priors) - 1), size=len(indices), p=[1.0 / (len(class_priors) - 1)] * (len(class_priors) - 1))
    for i, idx in enumerate(indices):
        component = mixture_component_labels[i]
        generated_samples[idx] = np.random.multivariate_normal(component_mean[component], component_covariance[component])
    return generated_samples, generated_labels

def generate_conf_matrix(predictions, labels, title): 
    confusion_matrix = skm.confusion_matrix(predictions, labels) 
    display_labels = [str(label) for label in np.unique(labels)]
    display = skm.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
    plt.figure()
    display.plot()
    plt.title(title)
    plt.show()

def data_visualization(predictions, labels, x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    marker_shapes = ['o', '^', 's', '.']
    for label, marker_shape in zip(np.unique(labels), marker_shapes):
        mask = labels == label
        ax.scatter(x[mask & (predictions != labels)], y[mask & (predictions != labels)], z[mask & (predictions != labels)], marker=marker_shape, color='red', label='Incorrect', s=10) 
        ax.scatter(x[mask & (predictions == labels)], y[mask & (predictions == labels)], z[mask & (predictions == labels)], marker=marker_shape, color='green', label='Correct', s=10)   
    ax.legend()
    plt.show()

if __name__ == "__main__":
    class_priors = np.array([0.3, 0.3, 0.4])
    mean = np.array([[0, 0, 0],[0, 0, 1]])    
    covariance = np.array([np.eye(3) * 0.5**2, np.eye(3) * 0.5**2])
    component_mean = np.array([[0, 0, 4 * 0.5], [0, 0, 6 * 0.5]])
    component_covariance = np.array([np.eye(3) * 0.5**2, np.eye(3) * 0.5**2])
    generated_samples, labels = generate_samples(mean, covariance, class_priors, 10000, component_mean, component_covariance)
    # Adjust the mean and covariance of the third class
    mean = np.array([mean[0], mean[1],(component_mean[0] + component_mean[1])/2])
    covariance = np.array([covariance[0], covariance[1], component_covariance[0]])

    # Data Visualization: scatter-plot in 3-dimensional space with different marker shape for each class
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_title("Distribution of Samples")
    marker_shapes = ['o', '^', 's', '.']
    for label in np.unique(labels):
        ax.scatter(generated_samples[:,0][labels == label], generated_samples[:,1][labels == label], generated_samples[:,2][labels == label], marker=marker_shapes[label], label=f'Class {label}')
        ax.legend()
    plt.show()

    # Part A: 0 - 1 Loss matrix 
    loss_mat_1 = np.ones((3, 3)) - np.eye(3)
    predictions = np.argmin(loss_mat_1.dot(np.diag(class_priors).dot(np.array([multivariate_normal.pdf(generated_samples, mean[i], covariance[i]) for i in range(len(mean))]))), axis=0)
    generate_conf_matrix(predictions, labels, '0 - 1 Loss Mat')
    data_visualization(predictions, labels, generated_samples[:,0], generated_samples[:,1], generated_samples[:,2], '0 - 1 Loss Mat')

    # Part B: Loss matrix with 10 and 100 times higher cost when Y = 3
    loss_mat_10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
    loss_mat_100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])
    predictions = np.argmin(loss_mat_10.dot(np.diag(class_priors).dot(np.array([multivariate_normal.pdf(generated_samples, mean[i], covariance[i]) for i in range(len(mean))]))), axis=0)
    generate_conf_matrix(predictions, labels, 'Loss Mat 10')
    data_visualization(predictions, labels, generated_samples[:,0], generated_samples[:,1], generated_samples[:,2], 'Loss Mat 10')

    predictions = np.argmin(loss_mat_100.dot(np.diag(class_priors).dot(np.array([multivariate_normal.pdf(generated_samples, mean[i], covariance[i]) for i in range(len(mean))]))), axis=0)
    generate_conf_matrix(predictions, labels, 'Loss Mat 100')
    data_visualization(predictions, labels, generated_samples[:,0], generated_samples[:,1], generated_samples[:,2], 'Loss Mat 100')
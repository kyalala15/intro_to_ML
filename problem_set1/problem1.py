import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
from sys import float_info  
from scipy.stats import multivariate_normal

# Generate samples from a Gaussian mixture model
def generate_samples(mean, covariance, class_priors, number_of_samples):
    dim = mean.shape[1]
    generated_samples = np.zeros([number_of_samples, dim])
    generated_labels = np.random.choice(len(class_priors), size=number_of_samples, p=class_priors)
    # Generate samples for each class based on the class priors and the class conditional distributions
    for k in range(len(class_priors)):
        indices = np.where(generated_labels == k)[0]
        num_samples_k = len(indices)
        generated_samples[indices, :] = multivariate_normal.rvs(mean[k], covariance[k], num_samples_k)
    return generated_samples, generated_labels

# Generate ROC curve from decision values and true labels for binary classification
def generate_roc(decision_values, true_labels):
    true_labels = np.array(true_labels)
    decision_values = np.array(decision_values)
    num_labels = np.array([np.sum(true_labels == 0), np.sum(true_labels == 1)])
    sorted_indices = np.argsort(decision_values)
    sorted_values = decision_values[sorted_indices]
    sorted_labels = true_labels[sorted_indices]
    thresholds = np.concatenate(([sorted_values[0] - float_info.epsilon], sorted_values, [sorted_values[-1] + float_info.epsilon]))
    TPR = []
    FPR = []
    # Calculate TPR and FPR for each threshold value in the sorted decision values array
    for threshold in thresholds:
        decisions = sorted_values >= threshold
        true_positives = np.sum((decisions == 1) & (sorted_labels == 1))
        false_positives = np.sum((decisions == 1) & (sorted_labels == 0))
        TPR.append(true_positives / num_labels[1])
        FPR.append(false_positives / num_labels[0])
    return np.array(FPR), np.array(TPR), thresholds

# Calculate binary decisions from predictions and true labels for binary classification
def calculate_binary_decisions(predictions, true_labels):
    true_negatives = np.sum((predictions == 0) & (true_labels == 0))
    false_positives = np.sum((predictions == 1) & (true_labels == 0))
    false_negatives = np.sum((predictions == 0) & (true_labels == 1))
    true_positives = np.sum((predictions == 1) & (true_labels == 1))
    total_negatives = np.sum(true_labels == 0)
    total_positives = np.sum(true_labels == 1)
    TNR = true_negatives / total_negatives if total_negatives != 0 else 0
    FPR = false_positives / total_negatives if total_negatives != 0 else 0
    FNR = false_negatives / total_positives if total_positives != 0 else 0
    TPR = true_positives / total_positives if total_positives != 0 else 0
    return TNR, FPR, FNR, TPR

if __name__ == "__main__":
    class_priors = np.array([0.7, 0.3])
    mean = np.array([[-1, 1, -1, 1],
                    [1, 1, 1, 1]])  
    covariance = np.array([[[2, -0.5, 0.3, 0],
                    [-0.5, 1, -0.5, 0],
                    [0.3, -0.5, 1, 0],
                    [0, 0, 0, 2]],
                    [[1, 0.3, -0.2, 0],
                    [0.3, 2, 0.3, 0],
                    [-0.2, 0.3, 1, 0],
                    [0, 0, 0, 3]]])  
    covariance_B = np.array([[[2, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 2]],
                    [[1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 3]]])

    # Generate samples and plot the distribution of samples in 3D space for the two classes 
    generated_samples, labels = generate_samples(mean, covariance, class_priors, 10000)
    samples_per_class = np.array([sum(labels == l) for l in np.array(range(2))])
    fig_distr = plt.figure()
    ax_distr = fig_distr.add_subplot(111, projection='3d')
    for i in np.array(range(2)):
        ax_distr.scatter(generated_samples[labels == i, 0], generated_samples[labels == i, 1], generated_samples[labels == i, 2], label="Class {}".format(i))
    plt.title("Distribution of Samples")
    plt.legend()
    plt.show()

    # Calculate the ROC curve for the generated samples and plot it 
    empirical_class_prob = np.array([multivariate_normal.pdf(generated_samples, mean[i], covariance[i]) for i in np.array(range(2))])
    empirical_gamma = np.log(empirical_class_prob[1]) - np.log(empirical_class_prob[0])
    fpr, tpr, empirical_gammas = generate_roc(empirical_gamma, labels)
    fig, ax = plt.subplots() 
    ax.plot(fpr, tpr, label="ROC Curve", color='b')
    ax.set_xlabel(r"False Positive Rate")
    ax.set_ylabel(r"True Positive Rate")

    # Calculate the minimum probability of error for the empirical classifier gamma thresholds
    empirical_pe = np.array((fpr, 1 - tpr)).T.dot(samples_per_class / 10000)
    theor_gamma = class_priors[0] / class_priors[1]
    map_ratio = empirical_gamma >= np.log(theor_gamma)
    tnr_theor, fpr_theor, fnr_theor, tpr_theor = calculate_binary_decisions(map_ratio, labels)
    min_pe = np.array((fpr_theor * class_priors[0] + fnr_theor * class_priors[1]))

    # Plot the minimum probability of error for the empirical classifier gamma thresholds
    ax.plot(fpr[np.argmin(empirical_pe)], tpr[np.argmin(empirical_pe)], 'mo', label="Empirical Minimum P[error]") 
    ax.plot(fpr_theor, tpr_theor, 'cd', label="Theoretical Minimum P[error] ") 
    
    # Label the minimum probability of error, gamma, false positive rate, and true positive rate for the empirical classifier gamma thresholds
    ax.annotate("Empirical Minimum P[error] = {:.3f}".format(np.min(empirical_pe)), (fpr[np.argmin(empirical_pe)], tpr[np.argmin(empirical_pe)]), textcoords="offset points", xytext=(10, 20), ha='center')
    ax.annotate("Empirical Minimum Gamma = {:.3f}".format(np.exp(empirical_gammas[np.argmin(empirical_pe)])), (fpr[np.argmin(empirical_pe)], tpr[np.argmin(empirical_pe)]), textcoords="offset points", xytext=(200, 20), ha='center')
    ax.annotate("FPR = {:.3f}".format(fpr[np.argmin(empirical_pe)]), (fpr[np.argmin(empirical_pe)], tpr[np.argmin(empirical_pe)]), textcoords="offset points", xytext=(330, 20), ha='center')
    ax.annotate("TPR = {:.3f}".format(tpr[np.argmin(empirical_pe)]), (fpr[np.argmin(empirical_pe)], tpr[np.argmin(empirical_pe)]), textcoords="offset points", xytext=(400, 20), ha='center')
    
    # Label the minimum probability of error, gamma, false positive rate, and true positive rate for the theoretical classifier gamma thresholds
    ax.annotate("Theoretical Minimum P[error] = {:.3f}".format(min_pe), (fpr_theor, tpr_theor), textcoords="offset points", xytext=(10, -40), ha='center')
    ax.annotate("Theoretical Minimum Gamma = {:.3f}".format(theor_gamma), (fpr_theor, tpr_theor), textcoords="offset points", xytext=(210, -40), ha='center')
    ax.annotate("FPR = {:.3f}".format(fpr_theor), (fpr_theor, tpr_theor), textcoords="offset points", xytext=(350, -40), ha='center')
    ax.annotate("TPR = {:.3f}".format(tpr_theor), (fpr_theor, tpr_theor), textcoords="offset points", xytext=(420, -40), ha='center')

    # Calculate the ROC curve for the naive classifier gamma thresholds
    naive_prob = np.array([multivariate_normal.pdf(generated_samples, mean[l], covariance_B[l]) for l in np.array(range(2))])
    naive_gamma = np.log(naive_prob[1]) - np.log(naive_prob[0])
    fpr_naive, tpr_naive, gammas_naive = generate_roc(naive_gamma, labels)
    pe_naive = np.array((fpr_naive, (1 - tpr_naive))).T.dot(samples_per_class / 10000)

    # Plot the ROC curve for the naive classifier gamma thresholds
    ax.plot(fpr_naive, tpr_naive, label="Naive ROC", linestyle='--', color='r')
    ax.plot(fpr_naive[np.argmin(pe_naive)], tpr_naive[np.argmin(pe_naive)], 'g^', label="Naive Minimum P[error]")

    # Label the minimum probability of error, gamma, false positive rate, and true positive rate for the naive classifier gamma thresholds
    ax.annotate("Naive Minimum P[error] = {:.3f}".format(np.min(pe_naive)), (fpr_naive[np.argmin(pe_naive)], tpr_naive[np.argmin(pe_naive)]), textcoords="offset points", xytext=(10, 0), ha='center')
    ax.annotate("Naive Minimum Gamma = {:.3f}".format(np.exp(gammas_naive[np.argmin(pe_naive)])), (fpr_naive[np.argmin(pe_naive)], tpr_naive[np.argmin(pe_naive)]), textcoords="offset points", xytext=(180, 0), ha='center')
    ax.annotate("FPR = {:.3f}".format(fpr_naive[np.argmin(pe_naive)]), (fpr_naive[np.argmin(pe_naive)], tpr_naive[np.argmin(pe_naive)]), textcoords="offset points", xytext=(300, 0), ha='center')
    ax.annotate("TPR = {:.3f}".format(tpr_naive[np.argmin(pe_naive)]), (fpr_naive[np.argmin(pe_naive)], tpr_naive[np.argmin(pe_naive)]), textcoords="offset points", xytext=(375, 0), ha='center')

    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.show()
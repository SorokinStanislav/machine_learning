import numpy as np
import matplotlib.pyplot as plt


# General function that compute binary confusion matrix based measures
from matplotlib.ticker import AutoMinorLocator


def binary_classification_measure(matrix, plot=False):
    true_negative = matrix[0][0]
    false_positive = matrix[0][1]
    false_negative = matrix[1][0]
    true_positive = matrix[1][1]
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    volume = positive + negative

    positive_part = positive / volume

    error_rate = (false_positive + false_negative) / (positive + negative)
    accuracy = (true_positive + true_negative) / (positive + negative)
    sensitivity = true_positive / positive
    specificity = true_negative / negative
    precision = true_positive / (true_positive + false_positive)
    fall_out = false_positive / (true_negative + false_positive)
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity)

    random_accuracy = ((true_negative + false_positive) * (true_negative + false_positive) +
                       (false_negative + true_positive) * (false_positive + true_positive)) / (volume * volume)
    kappa = (accuracy - random_accuracy) / (1 - random_accuracy)

    if plot:
        def f_score(beta):
            return ((1 + beta * beta) * (precision * sensitivity)) / (beta * beta * precision + sensitivity)

        beta_set = np.arange(0, 10.5, 0.5)
        plt.figure(2)
        ax = plt.axes()
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot(beta_set, f_score(beta_set))
        plt.xlabel('beta')
        plt.ylabel('F-score(beta)')
        plt.show()

        print(matrix)
        print("real positive: " + positive.astype('str'))
        print("real negative: " + negative.astype('str'))
        print("error rate: " + error_rate.astype('str'))
        print("accuracy: " + accuracy.astype('str'))
        print("sensitivity: " + sensitivity.astype('str'))
        print("specificity: " + specificity.astype('str'))
        print("precision: " + precision.astype('str'))
        print("fall-out: " + fall_out.astype('str'))
        print("f1 score: " + f1_score.astype('str'))
        print("Cohen's kappa: " + kappa.astype('str'))

        print('------------------------------------------------')

    return {'volume': volume, 'positive_part': positive_part, 'error_rate': error_rate, 'accuracy': accuracy,
              'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'fall_out': fall_out,
              'f1_score': f1_score, 'kappa': kappa}

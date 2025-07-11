'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    return np.sum(np.abs(Rx - Ry)) / len(Rx)


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    Lxx = np.sum((Thetax - np.sum(Thetax) / len(Thetax)) ** 2)
    Lyy = np.sum((Thetay - np.sum(Thetay) / len(Thetay)) ** 2)
    Lxy = np.sum((Thetax - np.sum(Thetax) / len(Thetax)) * (Thetay - np.sum(Thetay) / len(Thetay)))

    return (1 - (Lxy * Lxy) / (Lxx * Lyy)) * 100

# Mean Squared Error (MSE) as additional comparison
def mseDistance(imgA, imgB):
    """
    Computes the mean squared difference between two equally sized images.
    param imgA: First image.
    param imgB: Second image.
    return: Mean squared error between the two images.
    Hint: 0 means identical, higher values indicate more differences.
    """
    if imgA.shape != imgB.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Calculate the squared difference and then the mean
    err = np.mean((imgA.astype("float") - imgB.astype("float")) ** 2)
    
    return err

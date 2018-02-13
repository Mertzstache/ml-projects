#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from gmm_est import gmm_est
from scipy.stats import norm as nrm


def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    file_path = sys.argv[1]

    # YOUR CODE FOR PROBLEM 3 GOES HERE



    #Class 1
    mu1 = [9.77488592, 29.58258718] #[9.77484565, 29.58255052]#
    sigmasq1 = [21.92280453, 9.78376963] #[21.92236858, 9.78401755]
    wt1 = [0.59765463, 0.40234537]#[0.59765267, 0.40234733]
    loglikelihoods1 = [-3461.1653887258053, -3459.0969879474351, -3458.7567842479284, -3458.7045897577232, -3458.6966814059551, -3458.6954856341231, -3458.6953049269432, -3458.6952776233534, -3458.6952734982792, -3458.6952728750625, -3458.6952727809207, -3458.6952727666876, -3458.6952727645439, -3458.6952727642119, -3458.6952727641674, -3458.6952727641624, -3458.6952727641533, -3458.6952727641624, -3458.6952727641619, -3458.695272764161]

    #Class 2
    mu2 = [-24.82275173, -5.06015828, 49.62444472] #[-24.82274148, -5.06014954, 49.62444469] #[-10.73516438, 49.73097592] 
    sigmasq2 = [7.94733541, 23.32266181, 100.0243375]#[7.94739402, 23.32255764, 100.02433854] #[100.89404292, 97.46191226] 
    wt2 = [0.20364946, 0.49884302, 0.29750752]#[0.20364978, 0.4988427, 0.29750752] #[0.70364496, 0.29635504]
    loglikelihoods2 = [-8677.8265994064077, -8652.7367253934644, -8652.5772297395342, -8652.5745390771554, -8652.574489586581, -8652.5744886693556, -8652.5744886523626, -8652.5744886520406, -8652.5744886520424, -8652.5744886520242, -8652.5744886520351, -8652.5744886520242, -8652.5744886520097, -8652.5744886520279, -8652.5744886520279, -8652.5744886520279, -8652.574488652026, -8652.5744886520242, -8652.5744886520242, -8652.5744886520242]


    test_data1, test_data2 = read_gmm_file(file_path)

    p1 = float(len(test_data1))/(len(test_data1) + len(test_data2))
    results1 = gmm_classify(test_data1, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    results2 = gmm_classify(test_data2, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)

    #print results1

    class1_data = []
    class2_data = []
    for elem in range(len(results1)):
        if results1[elem] == 1:
            class1_data.append(test_data1[elem])
        else:
            class2_data.append(test_data1[elem])

    for elem in range(len(results2)):
        if results2[elem] == 1:
            class1_data.append(test_data2[elem])
        else:
            class2_data.append(test_data2[elem])


    #print len(class1_data)
    #print len(class2_data)
    # class1_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 1.
    print 'Class 1'
    print class1_data
    #print "for class1, we got a total of ", percent_correct(test_data1, class1_data), "percent correct"
    # class2_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 2.
    print '\nClass 2'
    print class2_data
    #print "we got a total of ", (percent_correct(test_data2, class2_data) + percent_correct(test_data1, class1_data)) /(len(test_data1) + len(test_data2)), "percent correct"
    #print "for class2, we got a total of ", percent_correct(test_data2, class2_data), "percent correct"



    #plotting!!

    # ones = [0 for i in range(len(class1_data))]
    # onesfortwo = [0 for i in range(len(class2_data))]

    # plt.hist(test_data1, 50,  alpha=0.5, facecolor='r')
    # plt.scatter(class1_data, ones, color='r', alpha=0.25)
    # plt.hist(test_data2, 50,  alpha=0.5, facecolor='b')

    # plt.scatter(class2_data, onesfortwo, color='b', alpha=0.25)

    # plt.title("Histogram of all data with classification by GMM \n (Red: Class 1, Blue: Class 2)")
    # plt.xlabel("Value of Data Points")
    # plt.ylabel("Frequency")
    # plt.tight_layout()
    # plt.savefig("classificationbyGMM.png")
    # plt.figure()


def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1): #1/3
    """
    Input Parameters:
        - X           : N 1-dimensional data points (a 1-by-N numpy array)
        - mu1         : means of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - sigmasq1    : variances of Gaussian components of the 1st class (a 1-by-K1 numpy array)
        - wt1         : weights of Gaussian components of the 1st class (a 1-by-K1 numpy array, sums to 1)
        - mu2         : means of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - sigmasq2    : variances of Gaussian components of the 2nd class (a 1-by-K2 numpy array)
        - wt2         : weights of Gaussian components of the 2nd class (a 1-by-K2 numpy array, sums to 1)
        - p1          : the prior probability of class 1.

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """

    class_pred = []
    for element in X:
        class1 = 0.0
        class2 = 0.0
        for gaussian in range(len(mu1)):
            class1 += p1*wt1[gaussian] * nrm(mu1[gaussian], np.sqrt(sigmasq1[gaussian])).pdf(element)
        for gaussian in range(len(mu2)):
            class2 += (1-p1)*wt2[gaussian] * nrm(mu2[gaussian], np.sqrt(sigmasq2[gaussian])).pdf(element)

        if class1 >= class2:
            class_pred.append(1)
        else:
            class_pred.append(2)


    # YOUR CODE FOR PROBLEM 3 HERE

    return class_pred


def percent_correct(originalData, asClassifiedData):
    correct = 0
    for classifiedData in originalData:
        if classifiedData in asClassifiedData:
            correct += 1
    return float(correct)/len(originalData)
def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()

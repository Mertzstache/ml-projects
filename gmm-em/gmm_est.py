#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as nrm


def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    file_path = sys.argv[1]

    # YOUR CODE FOR PROBLEM 2 GOES HERE

    examples1, examples2 = read_gmm_file(file_path)



    mu_results1, sigma2_results1, w_results1, L_results1 =  gmm_est(examples1, np.array([10,30]), np.array([10,10]), np.array([.5,.5]), 20) #gmm_est(examples, np.array([10,30]), [15,15], [0.5, 0.5], 20)
    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_results1, '\nsigma^2 =', sigma2_results1, '\nw =', w_results1
    #print 'log likelihoods: ', L_results1


    mu_results2, sigma2_results2, w_results2, L_results2 = gmm_est(examples2, np.array([-25,-1, 40]), np.array([7,23,100]), np.array([.3,.3,.4]), 20)    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu_results2, '\nsigma^2 =', sigma2_results2, '\nw =', w_results2
    #print 'log likelihoods: ', L_results2


    plt.title("Log Likelihoods for Class1")
    plt.xlabel("Iteration Number")
    plt.ylabel("Log Likelihood for that iteration")
    plt.scatter(range(1, len(L_results1)+1), L_results1)
    plt.xticks(range(1, len(L_results1)+1))
    plt.tight_layout()
    plt.savefig("likelyhood_class1.png")
    plt.figure()


    plt.title("Log Likelihoods for Class2")
    plt.xlabel("Iteration Number")
    plt.ylabel("Log Likelihood for that iteration")
    plt.scatter(range(1, len(L_results2)+1), L_results2)
    plt.xticks(range(1, len(L_results2)+1))
    plt.tight_layout()
    plt.savefig("likelyhood_class2.png")
    plt.figure()


def gmm_est(X, mu_init, sigmasq_init, wt_init, its):
    """
    Input Parameters:
      - X             : N 1-dimensional data points (a 1-by-N numpy array)
      - mu_init       : initial means of K Gaussian components (a 1-by-K numpy array)
      - sigmasq_init  : initial  variances of K Gaussian components (a 1-by-K numpy array)
      - wt_init       : initial weights of k Gaussian components (a 1-by-K numpy array that sums to 1)
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : means of Gaussian components (a 1-by-K numpy array)
      - sigmasq       : variances of Gaussian components (a 1-by-K numpy array)
      - wt            : weights of Gaussian components (a 1-by-K numpy array, sums to 1)
      - L             : log likelihood
    """


    numGauss = len(mu_init)
    numTrain = len(X)

    tolerance = 0.001
    mu = np.array(mu_init)
    sigmasq = np.array(sigmasq_init)
    wt = np.array(wt_init)
    gammas = np.zeros(len(mu_init))

    L = []

    for iteration in range(its):
    	#print iteration

    	#E STEP
    	lilgamma = np.zeros((numGauss, numTrain))
    	for i in range(numTrain):
    		# print lilgamma
    		total_probability_for_all_gaussians = 0.0
    		for j in range(numGauss):
    			current_probability = wt[j] * nrm(mu[j], np.sqrt(sigmasq[j])).pdf(X[i])
    			lilgamma[j, i] = current_probability
    			total_probability_for_all_gaussians += current_probability

    		if total_probability_for_all_gaussians > 0:
    			lilgamma[:,i] /= total_probability_for_all_gaussians



    	#print lilgamma
    	#M STEP

    	biggamma = np.zeros(numGauss)
    	for i in range(numGauss):
    		for j in range(numTrain):
    			biggamma[i] += lilgamma[i, j]
    	#print biggamma


    	wt = np.zeros(numGauss)
    	#previous_means = mu[:]
    	mu = np.zeros(numGauss)
    	sigmasq = np.zeros(numGauss)

    	for i in range(numGauss):

    		wt[i] = biggamma[i]/float(numTrain)

    		for j in range(numTrain):
    			#print lilgamma[i, j]
    			mu[i] += lilgamma[i, j]*X[j]

    		if biggamma[i] > 0:
    			mu[i] /= biggamma[i]
    		#print mu


    		#print wt
    		for j in range(numTrain):
    			sigmasq[i] += lilgamma[i, j]*np.square(X[j] - mu[i])


    		if biggamma[i] > 0:
    			sigmasq[i] /= biggamma[i]
    		#print sigmasq
    	#ll stuff

    	addL = 0.0
    	for i in range(numTrain):
    		prob = 0
    		for j in range(numGauss):
    			prob += wt[j]*nrm(mu[j],  np.sqrt(sigmasq[j])).pdf(X[i])
    		if prob > 0:
    			addL += np.log(prob)
    	# if (np.abs(ll_new - L)) < tolerance:
    	# 	break

    	L.append(addL)# = ll_new



    return mu, sigmasq, wt, np.array(L)


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

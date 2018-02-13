#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt


def build_x_matrix(X, k): 
	#small subroutine that builds a matrix using our x vector and a max polynomial. We go all the way up to k+1 because we want to include k
	#powers is just a list of the values of k, not necessary, but I thought it made things more clear. 
	powers = range(k+1)
	matrix = [ [] for i in range(len(X))] #initialization
	for i in range(len(X)): #for every example in X
		for pwr in powers: #append the value of i^pwr to the matrix we are returning
			matrix[i].append(np.power(X[i], pwr))
	return np.matrix(matrix) #actually returning the matrix 


def fold_data(X, Y, n): #we need to fold the data randomly in such a way that we can index both X and Y and find the same value at the same index
	temp_x = X[:] #instead of importing copy and using a deepcopy, just retrieve the list using this slicing
	temp_y = Y[:] 
	split_X = [[] for x in range(n)]
	split_Y = [[] for x in range(n)]
	for i in range(n):
		for j in range(len(X)/n): #truncate/round down because we can just add extra ones on at the end
			random = np.random.randint(len(temp_x)) #finidng a random element
			split_X[i].append(temp_x[random]) #appending that random element and its corresponding y value to y
			split_Y[i].append(temp_y[random])
			temp_x = np.delete(temp_x, random) #lets remove that from the list, so there are no repeats!
			temp_y = np.delete(temp_y, random)
	if (len(temp_x) > 0): #do we have any leftovers?
		for i in range(len(temp_x)): #if so, just add them in order to x and y. 
			split_X[i].append(temp_x[i])
			split_Y[i].append(temp_y[i])
	return split_X, split_Y #this is returning a list of lists, an even subset in every element without repeats. 

def initialize(data, testNum): # a small initialization subroutine to separate the training and testing data from our folded data. 
	test = [] 
	train = []
	for s in range(len(data)): #iterate over all data
		if (s == testNum): #if this is the index of the testing data, append each of the elements in that index to test. 
			for i in data[s]:
				test.append(i)
		else: #otherwise it must be part of our training data, so we append all of its elements to the testing list. 
			for i in data[s]:
				train.append(i)
	return train, test #returning the lists

def nfoldpolyfit(X, Y, maxK, n, verbose):

#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Bryan Pardo (This is where you put your name)
#


#above are the given comments. 


	#k_vals is where we are holding the current value of the polynomial we are looking at
	#we will be using this as a reference point throughout the code

	#mean_error is an array in which we are storing our values for our mean error for every value of k
	#we use it by adding every error and dividing by the total times we tested (which is just the length of x)

	# split_data_x and y are the folded versions of the data. 
	# we will split these into train_data_x and y 
	# and test_data_x and y to be able to separate them into the
	# current training and testing set.

	k_vals = range(maxK) #list of all values of k
	mean_error = [0 for i in range(maxK)] #setting up our mean error array
	split_data_x, split_data_y = fold_data(X, Y, n)#splitting data into folds


	for i in range(maxK):
		for test_me in range(len(split_data_x)):
			train_data_x, test_data_x = initialize(split_data_x, test_me) #we are on test number test_me, so we test on that fold, 
			train_data_y, test_data_y = initialize(split_data_y, test_me) #but the rest is training data
			xMatrix = build_x_matrix(train_data_x, i) #make dat matrix! we use the current training data and the currnent polynomial i
			#the following calculation is the closed-form solutoin defined in the slides.
			#I know that i could have used polyfit and gotten the same result, but this is what I thought of first. 
			f = np.poly1d(xMatrix.transpose().dot(xMatrix).getI().dot(xMatrix.transpose()).dot(train_data_y).getA1()[::-1]) 
			for j in range(len(test_data_y)):
				x_test = test_data_x[j] #ok so now that weve gotten our funciton, its time to test!
				y_test = test_data_y[j]
				mean_error[i] += np.square(test_data_y[j] - f(test_data_x[j])) #just adding the squared error for each training example
		#weve trained and tested for all n folds!! this means weve tested every single data member
		#i.e. the number of tests is len(X) so we divide by that to get the average 
		mean_error[i] = float(mean_error[i])/len(X)


	if (int(verbose) == 1): #if we want to output

		xMatrix = build_x_matrix(X, mean_error.index(min(mean_error))) #make dat matrix with the smallest error polynomial
		f = np.poly1d(xMatrix.transpose().dot(xMatrix).getI().dot(xMatrix.transpose()).dot(Y).getA1()[::-1]) #fmla again
		x_line = np.linspace(min(X), max(X), 500)# making a continuous line space that we can plot
		y_line = f(x_line) #y_function is a function of x_line!! python is cool like this ;)
		plt.plot(X, Y, 'o', x_line, y_line) #plotting the X Y and the x_new and y_new
		ax = plt.gca()
		fig = plt.gcf()
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("regression function for polynomial (%d) \n with minimum average mse" %mean_error.index(min(mean_error))) 

		plt.figure() #plotting mean squared error here!!
		plt.title('mean squared error as a function of polynomial k')
		plt.xlabel('value of k')
		plt.ylabel('mean squared error')
		plt.plot(range(maxK), mean_error)

		plt.show()


	return 0


def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = ('true' == sys.argv[4].lower() or '1' == sys.argv[4]) #bool(sys.argv[4]) #this doesnt work!! now we can also use upper and lower case true/false

	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)

	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()

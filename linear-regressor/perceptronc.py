import sys
import csv
import numpy as np
import scipy

def make_array(X): #lets make a 2d array with all three dimensions of x represented
	matrix = [ [] for i in range(len(X))]
	for elem in range(len(X)):
		matrix[elem].append(1)
		matrix[elem].append(X[elem])
		matrix[elem].append(np.square(X[elem]-1)) #here we added a third dimension of (xi-1)^2. we shift, but we want to keep a positive sign.
	return matrix

def classify(w, x): #we manually do the math for the classification. again, we are classifying based on the sign. 
	if((w[0]*x[0] + w[1]*x[1] + w[2]*x[2]) > 0):
		return 1
	else:
		return -1

def perceptronc(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.


	#basically we go through this data set as many tiems as it takes to get a perfect result. However, if it goes over
	#the limit we have set, just exit and give the result. we count this in epochs, i.e. one iteration through all the data.

	#we use an array with THREE elements because we are in three dimensions now!
	#we shift by either 1 or -1 based on which way we erred on
	#we also scale the slope by the value of x multiplied by the actual classification.
	#but now we also have to scale the third weight with the corresponding x2
	w = w_init[:]
	k = 0
	counter = 0
	x_array = make_array(X)
	e = 0

	while (counter < 999999):
		k = (k+1)%len(X)
		if (Y[k] != classify(w,x_array[k])): #if our classificatoin is incorrect, adjust!
			w[0] = w[0] + x_array[k][0]*Y[k]
			w[1] = w[1] + x_array[k][1]*Y[k]
			w[2] = w[2] + x_array[k][2]*Y[k]
		counter += 1
		if (k == 0): #we've iterated through all the data. 
			e += 1
			truecount = 0
			test_array = make_array(X)
			for entry in range(len(X)): #check to see if we got it all right!
		 		if(Y[entry] == classify(w, test_array[entry])):
		 			truecount+=1
		 	if (truecount == len(X)):
		 		break
	return (w, e)


def main():
	rfile = sys.argv[1]
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	w_init = [0,0,0]# INTIALIZE W_INIT
	w1, e1 = perceptronc(w_init, X1, Y1)
	w2, e2 = perceptronc(w_init, X2, Y2)

	print "number of epochs needed to classify first set ", e1
	print "number of epochs needed to classify second set ", e2

if __name__ == "__main__":
	main()

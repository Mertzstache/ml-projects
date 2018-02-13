import sys
import csv
import numpy as np
import scipy

def make_array(X): #lets make a 2D array!!
	matrix = [ [] for i in range(len(X))] #this matrix will have all 1s in the first column, then the values of x in the second column.
	for elem in range(len(X)):
		matrix[elem].append(1)
		matrix[elem].append(X[elem])
	return matrix #woo that was easy!

def classify(w, x): #we use this as a way to see whether a value should be classified as +1 or -1 based on its sign. 
	if((w[0]*x[0] + w[1]*x[1]) > 0): #manually doing the matrix multiplication!! 
		return 1
	else:
		return -1

def perceptrona(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.

	#basically we go through this data set as many tiems as it takes to get a perfect result. However, if it goes over
	#the limit we have set, just exit and give the result. we count this in epochs, i.e. one iteration through all the data.

	#we use an array with two elements because we are using y = w0 + w1x1 as our equation. (2 dimensions)
	#we shift by either 1 or -1 based on which way we erred on
	#we also scale the slope by the value of x multiplied by the actual classification.
	w = w_init[:] #making a copy of w_init
	k = 0
	counter = 0
	x_array = make_array(X) #here we are putting the data into teh correct format for our calculations
	e = 0

	while (counter < 99999): #do the following for a while
		k = (k+1)%len(X) #be in the data set!!
		if (Y[k] != classify(w,x_array[k])): 
			w[0] = w[0] + x_array[k][0]*Y[k]
			w[1] = w[1] + x_array[k][1]*Y[k]
		counter += 1
		if (k == 0): #test every epoch!
			e += 1 
			truecount = 0
			test_array = make_array(X) #test 
			for entry in range(len(X)):
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
	w_init = [0,0]# INTIALIZE W_INIT

	w1, e1 = perceptrona(w_init, X1, Y1)
	w2, e2 = perceptrona(w_init, X2, Y2)
	

	print "number of epochs needed to classify first set ", e1
	print "number of epochs needed to classify second set ", e2

if __name__ == "__main__":
	main()

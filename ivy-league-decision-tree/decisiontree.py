#Machine Learning hw1
import csv
import argparse
import numpy as np
import random

#streaming data in from the input file
def init_data(inputFileName):
	data = []
	with open(inputFileName) as csvfile: #opening the file as a csv
		test = csv.reader(csvfile, delimiter='\t')
		first_line = next(test) #keeping track of the attributes
		for row in test: #rest of the data gets classified into a dictoinary using the first line as the keys
			entry = dict(zip(first_line, row))
			data.append(entry)
	return data, first_line
	
def prior_prob(ls): #quick subroutine to calculate the overall probability of a true classification and give their respective string to compare it t
	count = 0
	for i in ls:
		if(i['CLASS'] == 'true'):
			count += 1
	if(float(count)/float(len(ls)) > 0.5):
		return 'true'
	else:
		return 'false'

def all_true(ls): #a short subtroutine to tell if the rest of the examples are all true
	sent = True
	for i in ls:
		if (i['CLASS'] == 'false'):
			sent = False
			break
	return sent

def all_false(ls): # a similar subroutine to tell if all teh examplse are all false
	sent = True
	for i in ls:
		if (i['CLASS'] == 'true'):
			sent = False
			break
	return sent

def entropy(examples): #the specific calculation of entropy
	count_true, count_false = count('CLASS', examples) #getting the count of numbers
	
	if (count_true < len(examples) and count_true > 0): # if they are all true or all false, return zero (if we didnt catch it before)
		return -float(count_true)/len(examples)*np.log2(float(count_true)/len(examples)) - (float(count_false)/len(examples))*np.log2(float(count_false)/len(examples)) #mathematics
	else:
		return 0 #returning zero otherwise
	

def gain(examples, attribute): #specific subroutine for calculating gain

	count_true, count_false = count(attribute, examples)
	examples_t = []
	examples_f = []
	for i in examples: #adding each example (based on their data) to a list where all the examples of a specific attribute are either true or false
		if(i[attribute] == 'true'):
			examples_t.append(i)
		else:
			examples_f.append(i)
	e = entropy(examples) #entropy of previous examples
	summation = (float(count_true)/len(examples))*entropy(examples_t) +  (float(count_false)/len(examples))*entropy(examples_f) #summation of the two possibilities
	return e - summation #finally subtracting the summation from the overall entropy

def choose_attribute(attributes, examples):
	maximum = 0
	max_index = 0
	for i in range(len(attributes)):
		if(gain(examples, attributes[i]) > maximum):

			maximum = gain(examples, attributes[i]) 
			max_index = i
	return attributes[max_index]

def count(attribute, examples): # a simple counter which counts the number of true and false instances of a certain attribute in a set of examples
	count_true = 0
	count_false = 0
	for i in examples:
		if (i['CLASS']=='true'):
			count_true += 1
		else :
			count_false += 1
	return count_true, count_false

def most_common_value(examples, attribute): #otherwise known as mode, gets the most common example, could be combined with count, but it was clearer for me to have them separate
	count_true = 0 
	count_false = 0
	for i in examples:
		if (i[attribute]=='true'): #accessing dictionary element "attribute" in all examples 
			count_true += 1
		else:
			count_false += 1
	if(count_true > count_false):
		return 'true'
	else:
		return 'false'

def ID3(examples, attributes, default, currentAttribute):
	if (len(examples) == 0): #if we have no examples left, then just add a leaf with the "default" example which is the most common classification
		return Node(default, currentAttribute, False, True)
	elif (all_true(examples)): #if all examples are true, then add a leaf classifying true
		return_me = Node('true', currentAttribute, (currentAttribute == 'root'), True)
		return return_me
	elif (all_false(examples)): #likewise with false
		return_me = Node('false', currentAttribute, (currentAttribute == 'root'), True)
		return return_me
	elif (len(attributes) == 0):#if we have no more attributes to examine just pick the most common classification from the current attribute
		return_me = Node(most_common_value(examples, currentAttribute), currentAttribute, False, True)
		return return_me
	else:
		best = choose_attribute(attributes, examples) #selecting the best attribute using the maximum gain greedy heuristic
		tree = Node(best, currentAttribute, False, False) #setting the tree root
		examples_t = []
		examples_f = []
		for i in examples: #counting the number of true and false examples and actually appening them to lists
			if(i[best] == 'true'):
				examples_t.append(i)
			else:
				examples_f.append(i)
		subtree_t = ID3(examples_t, remove_attribute(best, attributes), most_common_value(examples,'CLASS'), best) #subtree for the "true" decision
		subtree_f = ID3(examples_f, remove_attribute(best, attributes), most_common_value(examples,'CLASS'), best) #subtree for the "false" decision

		tree.trueChild = subtree_t #adding those subtrees as children to the new root
		tree.falseChild = subtree_f

		return tree



def train_test(trainingSetSize, data): #gives us a training and test set
	train_data = []
	test_data = []
	tester = random.sample(range(len(data)), trainingSetSize) #getting a random sample of ints of size trainingSetSize from the total number of data entries
	for i in range(len(data)): #adding the indicies selected to train_data and the rest to test_data
		if (i in tester):
			train_data.append(data[i])
		else:
			test_data.append(data[i])
	return train_data, test_data

def remove_attribute(attribute, ls): #a quick subroutine for removing a specific attribute from a list. 
	newls = []
	for i in ls:
		if (i != attribute):
			newls.append(i)
	return newls


def main_helper(inputFileName, trainingSetSize, numberOfTrials, verbose): #a helper function to make things more clear
	
	mean_id3 = 0
	mean_prior = 0
	trial_num = 0
	while(trial_num < int(numberOfTrials)): #starting from trial 0 to trial numberOfTrials
		print "\nthis is trial number ", trial_num, "\n_____________________________________"

		data, first_line = init_data(inputFileName) #sorting through raw data
		first_line = remove_attribute('CLASS',first_line) #attributes
		train, test = train_test(int(trainingSetSize), data) #setting the training adn testing data

		pprob_result = prior_prob(train) #probability

		tree = ID3(train, first_line, 'false', 'root')#initializing/training the tree 
		tree.output() #outputting tree

		
		
		id3_percent = 0
		prior_percent = 0


		for d in test:
			if (d['CLASS'] == check_value(tree, d)): #using a method to go through the tree by using a specific example and seeing if the classification is right.
				id3_percent +=1
			if (d['CLASS'] == pprob_result):
				prior_percent += 1



		id3_percent= float(id3_percent)/len(test) #scaling to 100%
		prior_percent= float(prior_percent)/len(test)

		mean_id3 += id3_percent*100
		mean_prior += prior_percent*100

		print "\n\nID3 percentage correct: ", id3_percent*100, "%  correct classification"
		print "prior probability correct: ", prior_percent*100, "%  correct classification"
		trial_num += 1
	if (int(verbose) == 1):
		print "example file used: ", inputFileName
		print "number of trials: ", int(numberOfTrials)
		print "training set size for each trial: ", int(trainingSetSize)
		print "testing set size for each trial: ", len(test)
		print "mean performance of decision tree over all trials: ", (float(mean_id3/int(numberOfTrials))), "%  correct classification"
		print "mean performance of prior probability over all trials: ", (float(mean_prior/int(numberOfTrials))), "%  correct classification"

	return 0


def check_value(node, ex): #chekcing value subroutine
	if(node.isLeaf or node.isRoot): #if the node is a leaf or a root then just return its attribute
		return node.attribute

	if ex[node.attribute] == 'true': #otherwise, if the example has a "true" for the certain attribute of the node, go down that path
		if (node.trueChild.isLeaf): #check before to make sure its not a leaf, dont want to go out of bounds!
			return node.trueChild.attribute
		else:
			return check_value(node.trueChild, ex) #ok definetly not a leaf, we are safe to recurse on it
	else:
		if (node.falseChild.isLeaf):
			return node.falseChild.attribute
		else:
			return check_value(node.falseChild, ex)


class Node: #node class!

	trueChild = None
	falseChild = None #nothing for now, but will get initialized
	def __init__(self, myAttribute, myParent, isRoot, isLeaf): #constructor
		self.parent = myParent
		self.attribute = myAttribute
		self.isRoot = isRoot
		self.isLeaf = isLeaf

	def output(self): # instaed of a toString method, i just made an output function to make it easier, it just prints the attributes of everything including the nodes relatives

		print "\nparent: ", self.parent,
		print "attribute: ", self.attribute,
		if(not self.isLeaf):
			if(self.trueChild.isLeaf):
				print "-", #print "trueChild: ", self.trueChild.attribute,
			else:
				print "trueChild: ", self.trueChild.attribute,
			if(self.falseChild.isLeaf):
				print "-", #print "falseChild: ", self.falseChild.attribute,
			else:
				print "falseChild: ", self.falseChild.attribute,
			if(not self.trueChild.isLeaf):
				self.trueChild.output()
			if(not self.falseChild.isLeaf):
				self.falseChild.output()
		
    	

if __name__ == "__main__": #the main function - i wrote it in such a way that allows us to write from the commnad line

	test_data = []
	parser = argparse.ArgumentParser()

	parser.add_argument('inputFileName')
	parser.add_argument('trainingSetSize')
	parser.add_argument('numberOfTrials')  
	parser.add_argument('verbose')

	arguments = parser.parse_args()

	main_helper(arguments.inputFileName, arguments.trainingSetSize, arguments.numberOfTrials, arguments.verbose)

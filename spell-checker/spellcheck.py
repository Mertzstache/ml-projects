#Machine Learning hw2
#!/usr/local/bin/python
import csv
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import pprint
import time





def divide_words(inputFileName):
	erroniousWords = []
	corrections = []
	with open(inputFileName, "U") as csvfile:
		test = csv.reader(csvfile, delimiter = "\t")
		for row in test:
			erroniousWords.append(row[0])
			corrections.append(row[1])
	#pp = pprint.PrettyPrinter()
	#pp.pprint(erroniousWords)
	#pp.pprint(corrections)
	csvfile.close()
	return erroniousWords, corrections

def get_list_of_words(inputFileName):
	data = []
	pp = pprint.PrettyPrinter()
	with open(inputFileName, "U") as csvfile:
		test = csv.reader(csvfile, delimiter = "\t")
		for row in test:
			#pp.pprint(row)
			for word in row:
				data.append(word)
	
	csvfile.close()
	#pp.pprint(data)
	return data


def measure_error(typos, truewords, dictionarywords):
	start = time.time()
	error_rate = 0
	ls = []
	for word in range(len(typos)):
		#print word
		myCorrection = find_closest_word(typos[word], dictionarywords)
		if (truewords[word] != myCorrection):
			error_rate += 1
			#print "incorrect classification of:", typos[word], ". supposed to be ",
			#print truewords[word], " but got classified as ", myCorrection
		ls.append(myCorrection)
	print time.time() - start


	with open('corrected.txt', 'wb') as csvfile:
	 	wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	 	wr.writerow(ls)

	return float(error_rate)/len(typos)

def qwerty_distance(char1, char2):
	char1 = char1.lower()
	char2 = char2.lower()
	if (char1 < 'a' or char1 > 'z' or char2 < 'a' or char2 > 'z'):
		return -1
	index1a = 0
	index2a = 0
	index1b = 0
	index2b = 0
	manhattan = [['q','w','e','r','t','y','u','i','o','p',],['a','s','d','f','g','h','j','k','l'], ['z','x','c','v','b','n','m']]
	founda = False
	foundb = False
	for row in range(len(manhattan)):
		if (not founda):
			for letter in range(len(manhattan[row])):
				if (char1 == manhattan[row][letter]):
					index1a = row
					index2a = letter
					founda = True
		if(not foundb):
			for letter in range(len(manhattan[row])):
				if (char2 == manhattan[row][letter]):
					index1b = row
					index2b = letter
					founda = True
		if(founda and foundb):
			break
	return abs(index1a - index1b) + abs(index1b - index2b)


def qwerty_levenshtein_distance(string1, string2, deletion_cost, insertion_cost):
#write some code to compute the levenshtein distance between two strings, given some costs, return the distance as an integer
	d = [[0 for j in range(len(string2)+1)] for i in range(len(string1)+1) ]
	for i in range(len(string1)):
		d[i][0] = i*deletion_cost
	for j in range(len(string2)):
		d[0][j] = j*insertion_cost
	for j in range(1, len(string2)):
		for i in range(1, len(string1)):
			if (string1[i] == string2[j]):
				d[i][j] = d[i-1][j-1] #no operation cost because they match
			else:
				#print string1[i], string2[j]
				d[i][j] = min([d[i-1][j] + deletion_cost, d[i][j-1] + insertion_cost, d[i-1][j-1] + qwerty_distance(string1[i], string2[j]) ])

	#pp = pprint.PrettyPrinter()
	#pp.pprint(d)
	#print d[len(string1)-1][len(string2)-1]
	return d[len(string1)-1][len(string2)-1]


def main_helper(ToBeSpellCheckedFileName, Dictionary):
	mistakes = get_list_of_words(ToBeSpellCheckedFileName)
	dictionary = get_list_of_words(Dictionary)
	#mistakes, truewords = divide_words(ToBeSpellCheckedFileName)
	#dictionary = get_list_of_words(Dictionary)
	corrections = []

	#totalcomp = len(mistakes)* len(dictionary)
	#print "total comparisons: ", totalcomp
	

	#qwerty_experiment(mistakes, truewords, dictionary)
	#error = measure_error(mistakes, truewords, dictionary)
	#print error


	for word in range(len(mistakes)):
	# 	print "chekcing ", mistakes[word]
	 	replacement = find_closest_word(mistakes[word], dictionary)
	# 	print "corrected to ", replacement
	 	corrections.append(replacement)

	with open('corrected.txt', 'wb') as csvfile:
	 	wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	 	wr.writerow(corrections)
	

def find_closest_word(string1, dictionary):
	#write some code to do this, calling levenshtein_distance, and return a string (the closest word)
	closest_word = ""
	distance = 90
	#print len(dictionary)
	for word in dictionary:
		dst = levenshtein_distance(string1, word, 1, 1, 1) 
		if (dst < distance):
			closest_word = word
			distance = dst
	return closest_word



def experiment_error(typos, truewords, dictionarywords, deletion_cost, insertion_cost, substitution_cost):
	start = time.time()
	error_rate = 0
	ls = []
	for word in range(len(typos)):
		myCorrection = experiment_find_closest_word(typos[word], dictionarywords, deletion_cost, insertion_cost, substitution_cost)
		if (truewords[word] != myCorrection):
			error_rate += 1
			#print "incorrect classification of:", typos[word], ". supposed to be ",
			#print truewords[word], " but got classified as ", myCorrection
		ls.append(myCorrection)
	print time.time() - start


	# with open('corrected.txt', 'wb') as csvfile:
	#  	wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	#  	wr.writerow(ls)

	return float(error_rate)/len(typos)

def experiment_find_closest_word(string1, dictionary, deletion_cost, insertion_cost, substitution_cost):
	#write some code to do this, calling levenshtein_distance, and return a string (the closest word)
	closest_word = ""
	distance = 90
	for word in dictionary:
		dst = levenshtein_distance(string1, word, deletion_cost, insertion_cost, substitution_cost) 
		if (dst < distance):
			closest_word = word
			distance = dst
	return closest_word

def experiment(typos, correct, dictionary):
	ls = []
	max_number = 3
	for i in range(1,max_number +1):
		for j in range(1,max_number +1):
			for k in range(1,max_number +1):
				ls.append([i, j, k, 0])
	#print ls[1][1]
	for test in range(len(ls)):
		ls[test][3] = experiment_error(typos, correct, dictionary, ls[test][0], ls[test][1], ls[test][2])
	pp = pprint.PrettyPrinter()
	pp.pprint(ls)


def qwerty_experiment(typos, correct, dictionary):
	ls = []
	max_number = 3
	for i in range(1,max_number +1):
		for j in range(1,max_number +1):
			ls.append([i, j, 0])
	#print ls[1][1]
	#print ls
	for test in range(len(ls)):
		ls[test][2] = qwerty_experiment_error(typos, correct, dictionary, ls[test][0], ls[test][1])
	pp = pprint.PrettyPrinter()
	pp.pprint(ls)

def qwerty_experiment_error(typos, truewords, dictionarywords, deletion_cost, insertion_cost):
	start = time.time()
	error_rate = 0
	ls = []
	for word in range(len(typos)):
		myCorrection = qwerty_experiment_find_closest_word(typos[word], dictionarywords, deletion_cost, insertion_cost)
		if (truewords[word] != myCorrection):
			error_rate += 1
			#print "incorrect classification of:", typos[word], ". supposed to be ",
			#print truewords[word], " but got classified as ", myCorrection
		ls.append(myCorrection)
	print time.time() - start


	# with open('corrected.txt', 'wb') as csvfile:
	#  	wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	#  	wr.writerow(ls)

	return float(error_rate)/len(typos)

def qwerty_experiment_find_closest_word(string1, dictionary, deletion_cost, insertion_cost):
	#write some code to do this, calling levenshtein_distance, and return a string (the closest word)
	closest_word = ""
	distance = 90
	for word in dictionary:
		#print "looking at ", string1, "current word ", word
		dst = qwerty_levenshtein_distance(string1, word, deletion_cost, insertion_cost) 
		if (dst < distance):
			closest_word = word
			distance = dst
	return closest_word

def levenshtein_distance(string1, string2, deletion_cost, insertion_cost, substitution_cost):
	#write some code to compute the levenshtein distance between two strings, given some costs, return the distance as an integer
	d = [[0 for j in range(len(string2)+1)] for i in range(len(string1)+1) ]
	for i in range(len(string1)):
		d[i][0] = i*deletion_cost
	for j in range(len(string2)):
		d[0][j] = j*insertion_cost
	for j in range(1, len(string2)):
		for i in range(1, len(string1)):
			if (string1[i] == string2[j]):
				d[i][j] = d[i-1][j-1] #no operation cost because they match
			else:
				d[i][j] = min([d[i-1][j] + deletion_cost, d[i][j-1] + insertion_cost, d[i-1][j-1] + substitution_cost])

	#pp = pprint.PrettyPrinter()
	#pp.pprint(d)
	#print d[len(string1)-1][len(string2)-1]
	return d[len(string1)-1][len(string2)-1]



if __name__ == "__main__":
	test_data = []
	parser = argparse.ArgumentParser(description='give me details for the thing')

	parser.add_argument('ToBeSpellCheckedFileName')
	parser.add_argument('Dictionary')

	arguments = parser.parse_args()

	main_helper(arguments.ToBeSpellCheckedFileName, arguments.Dictionary)

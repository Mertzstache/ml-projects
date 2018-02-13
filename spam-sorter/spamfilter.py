#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil


def pprob(word, sham_list, directory):
	total_prob = 1
	for sh in sham_list:
		if word in parse(open(directory + sh)):
			total_prob += 1
	return float(total_prob)/float(len(sham_list)+1)

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
	#Making the dictionary. 
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	if ".DS_Store" in spam:
		spam.remove(".DS_Store")
	if ".DS_Store" in ham:
		ham.remove(".DS_Store")
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	words = {}

	#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
	for s in spam:
		#print "file: ", spam_directory + s
		for word in parse(open(spam_directory + s)):
			if word not in words:
				words[word] = {'spam': pprob(word, spam, spam_directory), 'ham': pprob(word, ham, ham_directory)}
	for h in ham:
		#print "file: ", ham_directory + h
		for word in parse(open(ham_directory + h)):
			if word not in words:
				words[word] = {'spam': pprob(word, spam, spam_directory), 'ham': pprob(word, ham, ham_directory)}
	
	#Write it to a dictionary output file.
	writedictionary(words, dictionary_filename)
	
	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.

	counter = 0
	nin = 0
	maxspam = np.log10(spam_prior_probability)
	maxham = np.log10(1 - spam_prior_probability)
	for word in content:

		if word not in dictionary:
			nin += 1
		if word in dictionary:
			if dictionary[word]['spam'] > dictionary[word]['ham']:
				counter += 1
			maxspam += np.log10(dictionary[word]['spam'])
			maxham += np.log10(dictionary[word]['ham'])
	if (maxspam >= maxham):
		return True
	else:
		return False


def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]

	if(".DS_Store" in mail):
		mail.remove(".DS_Store")
	for m in mail:
		content = parse(open(mail_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam:
			shutil.copy(mail_directory + m, spam_directory)
		else:
			shutil.copy(mail_directory + m, ham_directory)

def readdict(dictionary_filename):
	d = {}
	skipped = False
	with open(dictionary_filename, "U") as dictionary:
		for row in dictionary:
			if not skipped:
				skipped = True
			else:
				row.strip("\n")
				key, spam, ham = row.split("\t")
				spam = float(spam)
				ham = float(ham)
				d[key] = { 'spam' : spam, 'ham' : ham }
	return d

def test_with_dictionary(dictionary, spamdir, hamdir):
	spam = [f for f in os.listdir(spamdir) if os.path.isfile(os.path.join(spamdir, f))]
	ham = [f for f in os.listdir(hamdir) if os.path.isfile(os.path.join(hamdir, f))]

	if(".DS_Store" in spam):
		spam.remove(".DS_Store")
	if(".DS_Store" in ham):
		ham.remove(".DS_Store")

	spam_correct = 0
	ham_correct = 0
	###TESTING SPAM EMAILS
	for s in spam:
		content = parse(open(spamdir + s))
		test = is_spam(content, dictionary, 0.2)
		if test:
			spam_correct += 1

	for h in ham:
		content = parse(open(hamdir + h))
		test = is_spam(content, dictionary, 0.2)
		if not test:
			ham_correct += 1

	#print "for this test, we got a total of ", spam_correct, " or", float(spam_correct)/len(spam), " spam emails correct, and "
	#print "a total of ", ham_correct, " or ", float(ham_correct)/len(ham),  " ham emails correct. "

def select_random(ls, n):
	newl = ls[:]
	return_me = []
	for i in range(n):
		added = np.random.randint(0,len(newl))
		return_me.append(newl[added])
		newl.remove(newl[added])
	return return_me

def copy_folders(spam_directory, ham_directory, n):
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]

	if ".DS_Store" in spam:
		spam.remove(".DS_Store")
	if ".DS_Store" in ham:
		ham.remove(".DS_Store")
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))


	for i in range(n):
		newspam = select_random(spam, 50)
		newham = select_random(ham, 250)
		words = {}
		#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
		for s in newspam:
			#print "file: ", spam_directory + s
			for word in parse(open(spam_directory + s)):
				if word not in words:
					words[word] = {'spam': pprob(word, newspam, spam_directory), 'ham': pprob(word, newham, ham_directory)}
		for h in newham:
			#print "file: ", ham_directory + h
			for word in parse(open(ham_directory + h)):
				if word not in words:
					words[word] = {'spam': pprob(word, newspam, spam_directory), 'ham': pprob(word, newham, ham_directory)}
		
		#Write it to a dictionary output file.
		filename = "dictionary"+str(i)+".dict"
		writedictionary(words, filename)
	

if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]
	
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	
	dictionary_filename = "dictionary.dict"
	

	#FOR TESTING::::::::::::::::
	# d1 = readdict("dictionaryfirsthalf.dict")
	# d2 = readdict("dictionarysecondhalf.dict")


	#TRAINNUM = 5
	#copy_folders(training_spam_directory, training_ham_directory, TRAINNUM)

	# for i in range(5):
	# 	filename = "dictionary" + str(i) + ".dict"
	# 	test_with_dictionary(readdict(filename),"spam/", "easy_ham/")
	#test_with_dictionary(d2, "folds/fold1/spam_half/", "folds/fold1/ham_half/") ##fold1 has the second half in it
	#test_with_dictionary(d1, "folds/fold2/ham_half_first/", "folds/fold2/spam_half_first/") ##fold2 has the first half in it
	#:::::::::::::::::::::::

	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability)
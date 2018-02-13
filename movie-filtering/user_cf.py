# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables.
# Do not change main() function,

# import modules you need here.
import sys
import scipy.stats
import csv

#scipy.stats.pearsonr(array1,array2)

def user_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    '''
    build user-based collaborative filter that predicts the rating
    of a user for a movie.
    This function returns the predicted rating and its actual rating.

    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <distance> - a Boolean. If set to 0, use Pearson's correlation as the distance measure. If 1, use Manhattan distance.
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering,
    only users that have actual (ie non-0) ratings for the movie are considered in your top K.
    For user-based, use only movies that have actual ratings by the user in your top K.
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.
    <numOfUsers> - the number of users in the dataset
    <numOfItems> - the number of items in the dataset
    (NOTE: use these variables (<numOfUsers>, <numOfItems>) to build user-rating matrix.
    DO NOT USE any CONSTANT NUMBERS when building user-rating matrix. We already set these variables in the main function for you.
    The size of user-rating matrix in the test case for grading could be different from the given dataset. )

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Bongjun Kim (This is where you put your name)
    '''

    data = helper(datafile)
    #print len(data)
    userVectors = get_user_vectors(data, numOfUsers, numOfItems)
    trueRating = userVectors[userid-1][movieid-1] #we shift down because the array starts at zero.
    if (distance == 0):
        predictedRating = pearsons(userVectors, userid, movieid, k, iFlag)
    elif(distance == 1):
        predictedRating = manhattan(userVectors, userid, movieid, k, iFlag)

    return trueRating, predictedRating


def pearsons(userVectors, userid, movieid, k, iFlag):
    #our pearsons function
    ls = []
    for otheruser in range(len(userVectors)):
        if (otheruser != userid-1):
            temp, pval = scipy.stats.pearsonr(userVectors[userid-1],userVectors[otheruser])
            ls.append((abs(temp),otheruser))
    return topk(ls, userVectors, userid, movieid, k, iFlag, 0)
def manhattan(userVectors, userid, movieid, k, iFlag):
    #our manhattan distance funciton
    ls = []
    for otheruser in range(len(userVectors)):
        if (otheruser != userid-1):
            temp = scipy.spatial.distance.cityblock(userVectors[userid-1],userVectors[otheruser])
            ls.append((abs(temp),otheruser))
    return topk(ls, userVectors, userid, movieid, k, iFlag, 1)


def length_of_nonzeros(SpiceGirls):
    counter = 0
    for integer in SpiceGirls:
        if (integer != 0):
            counter+=1
    return counter

def topk(ls, userVectors, userid, movieid, k, iFlag, distance):
    '''
    Basically here we are choosing the top k
    if we pass an iFlag of 0, meaning we dont want to consider nonexistent reviews
    then we dont increment i, take the maximal elemnt out of our temporary list and we continue on our way
    if we run out, then just exit with the mode of whatever elements we did find.
    temporary is a copy of ls
    klist is the raw data of teh nearest k neighbors
    survey is the parsed klist and we take the mode of it to get a result
    '''
    temporary = ls[:]
    survey = []
    klist = []
    i = 0
    sent = True

    if(length_of_nonzeros([j[movieid-1] for j in userVectors]) == 0):
        return 0

    while (i < k and sent):
        if(len(temporary) == 0):
            sent = False
            print "with userid: ", userid, " and movie id: ", movieid
            print "we have a case where there are no more users left reviwing this movie."
            print "this means that this should return a number below k: ", length_of_nonzeros([j[movieid-1] for j in userVectors])-1
            print "with value of k: ", k
            print "therefore we just exit with whatever values we have, and see what the mode is from there"
            break
        if(distance == 1):
            temp = min([j[0] for j in temporary])
        else:
            temp = max([j[0] for j in temporary])
        index = [j[0] for j in temporary].index(temp)
        if(iFlag == 0):
            if(userVectors[temporary[index][1]][movieid-1] != 0):
                klist.append(temporary[index])
                i+=1
        else:
            klist.append(temporary[index])
            i+=1
        del temporary[index]
    for i in range(len(klist)):
        survey.append(userVectors[klist[i][1]][movieid-1])
    a, b = scipy.stats.mode(survey)
    return a[0]

def helper(inputFileName): #returns a vector with each user being the index and the movie numbers they review, score appended
	return_me = [[] for i in range(943)]
	with open(inputFileName, "U") as csvfile:
		test = csv.reader(csvfile, delimiter = "\t")
		for row in test:
			#print int(row[0])
			return_me[int(row[0])-1].append([int(row[1]), int(row[2])])
	csvfile.close()
	return return_me

def get_user_vectors(datafile, numOfUsers, numOfItems):
    return_me = [[0 for i in xrange(numOfItems)] for j in xrange(numOfUsers)]
    #print len(return_me)
    for i in range(numOfUsers):
        for j in datafile[i]:
            #print j
            return_me[i][j[0]-1] += j[1] #shift down because its smaller
    return return_me


def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682

    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()

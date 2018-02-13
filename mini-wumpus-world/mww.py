#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mini Wumpus World Thing"""

#****************************************
# MWW.py
# What this script does is calculates the q values for every pair
# and prints them out using a pretty printer
# it prints them out in order of the squares, starting from 1 (a1), to 9 (c3)
# you run the program like this:
# $ python mww.py
#****************************************

def getdirection(state1, state2):
    """gets direction, 0: up, 1: down, 2: left, 3: right -1: error"""
    if abs(state1 - state2) < 2:
        if state2 > state1:
            return 3
        return 2
    else:
        if state2 > state1:
            return 1
        return 0



def miniwumpus(world, actions, alpha, gamma, rewardmap):
    """edits world by reference"""
    for state in range(len(actions) - 1):
        #subroutine that gets the direction in terms of the index of the table
        direction = getdirection(actions[state], actions[state+1])
        #now we get the reward from the reward map
        reward = rewardmap[actions[state+1]]
        #and do our calculations by reference so that the actual table gets edited.
        world[actions[state]][direction] = (1 - alpha)*world[actions[state]][direction] + alpha*(reward + gamma*max(world[actions[state+1]]))


def main():
    """main function"""
    adventures = [[6, 3, 4], [6, 3, 0, 1, 2], [6, 7, 4], [6, 7, 8, 5, 2], [6, 7, 8, 5, 4]] #adventures
    world = [[0 for k in range(4)] for j in range(9)] #world map of q value pairs
    printhelper = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    worldmap = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    rewardmap = [0, 0, 50, 0, -10, 0, 0, 0, 0]

    #actual calculation by passing by reference
    for adventure in adventures:
        miniwumpus(world, adventure, 0.5, 0.5, rewardmap)


    #printing!
    print "map of world to these squares: "
    for line in printhelper:
        print line
    print "square number: values [up, down, left, right] "
    for line in range(len(world)):
        print "square:", line+1, "with values", world[line]


if __name__ == "__main__":
    main()

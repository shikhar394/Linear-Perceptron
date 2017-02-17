# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:45:24 2017

@author: shikhar
Splits the training file into training and validation set.
"""
TRAINING_SIZE = 4000

def splitDataSet(File):
    count = 1
    file = open(File, 'r')
    train = open("train.txt", 'w')
    validate = open("validate.txt", 'w')
    for line in file:
        line = line.strip()
        if count <= TRAINING_SIZE:
            train.write(line + '\n')
            count += 1
        else:
            validate.write(line + '\n')
            count += 1
    file.close()
    train.close()
    validate.close()

if __name__ == "__main__":
    splitDataSet("spam_train.txt")


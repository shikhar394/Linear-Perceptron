"""
Created on Mon Feb  6 19:33:00 2017

@author: Shikhar Sakhuja

Linear Perceptron to classify Spam from Non Spam.
Default configuration:
    spam_train.txt split into 4000 training mails and 1000 validation mails
    Global constants declared.

Trained on the basis of:
	X[i] = Feature Vector of the i(th) email
	Y[i] = Label of the i(th) email designating it as spam and non-spam. 
	W = initialized as 0's to be trained according to the algorithm

	if Y[i] * (W . X[i]) <= 0:
		W = W + Y[i] * X[i]

After training the weight using the algorithm, future emails can be classified as:
	W . X[i] > 0 ==> Spam
	W . X[i] < 0 ==> Non-Spam
	where W is the trained weights from the training set
	X[i] is the feature vector of the email
"""
import numpy as np
import matplotlib.pyplot as plt
import operator
from math import inf

SPAM = 1 #Representation of spam in mails
NOT_SPAM = 0 #Representation of non-spam in mails
SPAM_LABEL = 1 #Representation of Spam in our Label set i.e. Y
NOT_SPAM_LABEL = -1 #Representation of Non-Spam in our Label set i.e. Y
DEFAULT_COMMON_WORDS = 20

def VocabWords(FileName, X = DEFAULT_COMMON_WORDS, Num_Emails = inf):
    """
    Defines words that are present more than X (Default = 20) times in mails of the given file.
    Allows us to define our Feature Vectors.
    In some cases accepts Number of emails as Num_Emails when experimenting with different training sizes.
    """
    File = open(FileName, 'r')
    Vocab = {}
    count = 0
    for Email in File:
        Email = Email.rstrip().split()
        Email = Email[1:]
        Email = set(Email)
        Email = list(Email)
        for word in Email:
            if word not in Vocab:
                Vocab[word] = 1
            else:
                Vocab[word] +=1
        count += 1
        if count == Num_Emails:
            break
    File.close()
    return [word for word in Vocab if Vocab[word]>=X]

def XY_Vectors(FileName, WordList, Num_Emails = inf):
    """
    Returns X and Y Vectors.
    Y is our Label.
    X is our Feature Vector for all the mails in our training sets.
    Allows us to train our perceptron.
    """
    TrainingFile = open(FileName, 'r')
    count = 0
    FeatureVector= []
    Label = []
    for email in TrainingFile:
        count += 1
        if email[0]== str(SPAM):
            Label.append(SPAM_LABEL)
        elif email[0] == str(NOT_SPAM):
            Label.append(NOT_SPAM_LABEL)
        email = email.split()
        email = email[1:]
        FeatureVector.append([1 if word in email else 0 for word in WordList])
        if count == Num_Emails:
            break
    Label = np.array(Label, dtype=np.int)
    FeatureVector = np.array(FeatureVector, dtype=np.int)
    TrainingFile.close()
    print("Length of Feature Vector {}, Feature Vector {}".format(len(FeatureVector), FeatureVector))
    return (FeatureVector, Label)


def Perceptron_Train(FileName, WordList, Max_Iterations = inf, Num_Emails = inf):
    """
    Calls XY_Vectors to get Feature Vectors and Label for mails. Trains Weights on the basis of X and Y.
    Accepts Max_iterations to stop after some particular iterations if provided.
    """
    print('Length', Num_Emails)
    X, Y = XY_Vectors(FileName, WordList, Num_Emails)
    W = np.zeros((len(X[0])), dtype=np.int) #Length of Features in Vector
    print("Length of Weight", len(X[0]))
    Converge = False
    Updates = 0
    iterations = 0
    while Converge != True:
        Converge = True
        iterations += 1
        for i in range(len(X)):
            if (Y[i] * (np.dot(W, X[i]))) <= 0:
                if sum(X[i]) == 0:
                    continue
                Updates += 1
                W = W + (Y[i] * X[i])
                Converge = False
        print("Iteration {} : {} Mistakes".format(iterations, Updates))
        if Max_Iterations == iterations:
            Converge = True
    return (W, iterations, Updates)


def Perceptron_Error(W, WordList, FileName):
    """
    Uses the W (Trained parameters) to classify the validation sets. Returns the percentage error of classification.
    """
    File = open(FileName, 'r')
    Validation_Data_Size = len(File.readlines())
    X, Y = XY_Vectors(FileName, WordList, Validation_Data_Size)
    print(Validation_Data_Size)
    Error = 0
    print('X', len(X))
    print("W", len(W))
    for i in range(len(X)):
        if (Y[i] * (np.dot(W, X[i]))) <= 0:
            Error +=1
    return (Error/Validation_Data_Size)*100

def Word_Classifier(W, WordList):
    """
    Returns the 12 Words most assosciated with Spam and NonSpam in a given training set.
    """
    print(len(W) == len(WordList))
    Word_Weight = {word:weight for (word, weight) in zip(WordList, W)}
    Word_Weight = sorted(Word_Weight.items(), key = operator.itemgetter(1))
    print(len(W) == len(WordList) == len(Word_Weight))
    return (Word_Weight[:12], (list(reversed(Word_Weight[-12:]))))

def Graph_Plotting(X, Xlabel, Y, Ylabel, color, name):
    """
    Function for plotting graphs.
    Used to map varying validation errors and iterations required to converge for different sizes of training set on a graph.
    """
    plt.plot(X, Y, color)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(name)
    #plt.axis([0, X[-1], 0, Y[-1]])
    plt.show()

def Varying_Training_Size(TrainingFile, ValidationFile, Num_Emails):
    """
    Plots the graph of varying validation errors and iterations for different sizes of Training sizes.
    """
    Errors = []
    Iterations_List = []
    for i in range(len(Sizes)):
        WordList_i = VocabWords(TrainingFile, Num_Emails = Sizes[i])
        W, Iterations, Updates = Perceptron_Train(TrainingFile, WordList_i, Num_Emails = Sizes[i])
        Errors.append(Perceptron_Error(W, WordList_i, ValidationFile))
        Iterations_List.append(Iterations)
        print("Iterations: {} ---> Errors: {}".format(Iterations_List, Errors))
    Graph_Plotting(Sizes, 'Size of Training Set', Errors, "Error on the corresponding training set", 'r', "Errors corresponding to varying training sizes")
    Graph_Plotting(Sizes, 'Size of Training Set', Iterations_List, "Iterations before converging", 'b', "Iterations before converging to varying training sizes")

if __name__ == '__main__':
    TrainingFile = 'train.txt'
    ValidationFile = 'validate.txt'
    Original_TrainFile = 'spam_train.txt'
    Original_TestFile = 'spam_test.txt'
    Sizes = [100, 200, 400, 800, 2000, 4000]
    print('-'*67)

    Default_WordList= VocabWords(TrainingFile)
    print("\nTraining Set = 4000 Emails, Validation = 1000, X = 20")
    Default_W, Default_Iterations, Default_Updates = Perceptron_Train(TrainingFile, Default_WordList)
    Default_Error_ValidationSet = Perceptron_Error(Default_W, Default_WordList, ValidationFile)
    print("Weight {} \nIterations {} \nUpdates {} \nErrorRate {}".format(Default_W, Default_Iterations, Default_Updates, Default_Error_ValidationSet))


    Error_TrainingSet = Perceptron_Error(Default_W, Default_WordList, TrainingFile)
    if Error_TrainingSet == 0:
        print("Perceptron classifies Training Set perfectly")

    print("-"*67)

    Max_NonSpam_Words, Max_Spam_Words = Word_Classifier(Default_W, Default_WordList)
    print('Words most representative of Spam: \n')
    for weight, word in Max_Spam_Words:
        print('{} : {} \n'.format(weight, word))
    print('Words most representative of NonSpam: \n')
    for weight, word in Max_NonSpam_Words:
        print('{} : {} \n'.format(weight, word))

    print("-"*67)

    Varying_Training_Size(TrainingFile, ValidationFile, Sizes)

    print("-"*67)
    print("Using the entire training file and the new validation file")
    WordList = VocabWords(Original_TrainFile, 30)
    W, Iterations, Updates = Perceptron_Train(Original_TrainFile, WordList, Max_Iterations = 11)
    Error_CompleteTraining = Perceptron_Error(W, WordList, Original_TestFile)
    print(Error_CompleteTraining)


import pandas as pd
import numpy as np
from Node import Node
from Train_data import Train_data
from prepare_data import prepare_data
from Test_data import Test_data

class Random_forest:
    
    '''
    The class Random_forest implements a reandom forest, made of multiple decision trees.
    Each Tree is trained on a different training set, though the training sets might share some datapoints (bag of features).
    The random forest classifies datapoints by looking at the classification from each tree 
    and then choosing the most common one.
    
    The Random forest is first trained with train() and then can be used to classify datapoints with 
    classifySet() or the accuracy can be calculated directly with accuracy()
    
    The prepare_data() function splits the data into a training and a testing set according to the testRatio and then
    samples the training chunks from the training set according to the trainRatio.
    
    The variable trainRatio specifies how large a training set is, in realtion to the wohle data set, 
    testRatio does the same for the testing set. nrTrees indicates how many decision trees are making up the forest.
    
    '''

    def __init__(self, data, target, trainRatio = 0.1, nrTrees = 10, testRatio = 0.3):

        self.data = data
        self.target = target
        self.trees = []
        self.trainRatio = trainRatio 
        self.nrTrees = nrTrees
        self.testRatio = testRatio
        self.testingSet = None

    
    def train(self):

        trainingSets = self.prepare_data()

        # train a tree for each training set
        for trainingSet in trainingSets:

            attributes = list(trainingSet.columns)
            attributes.remove(self.target)

            rootNode = Node()
            decisionTree = Train_data(data=trainingSet, target=self.target, attributes=attributes, node=rootNode, max_recursion = np.PINF)
            decisionTree.id3()
            self.trees.append(rootNode)
        
    def prepare_data(self):
    
        # check whether Ratios are acceptable
        if self.testRatio >= 1:
            raise TypeError("Testing ratio has to be smaller than 1")
            
        if self.trainRatio >= 1:
            raise TypeError("Training ratio has to be smaller than 1")
        
            
        # remove any Nans from Dataframe
        data = self.data.dropna(how='any')

        # shuffle data
        data = data.sample(frac=1, random_state=1).reset_index(drop=True)
        
        # split of the testing set
        self.testingSet = data.iloc[0:int(len(data) * self.testRatio), :]
        trainingData = data.iloc[int(len(data) * self.testRatio):, :]

        # split training Data into random sets
        trainingSize = int(self.trainRatio * len(data))
        trainingSets = []
        for tree in range(self.nrTrees):
            # suffel dataset before extracting traingset of correct size
            trainingData = trainingData.sample(frac=1, random_state=1).reset_index(drop=True) #random shuffel
            trainingSets.append(trainingData.iloc[:trainingSize, :])
        
        return trainingSets
    
    def classifySet(self, testingSet):
        
        # for each tree get classification list
        classifications = []
        for tree in self.trees:
            testData = Test_data(testingSet, self.target, tree)
            classifications.append(testData.classifySet())
        
        # for each datapoint get the most common classification
        votedClass = []
        for i in range(len(testingSet)):
            votes = []
            for classList in classifications:
                votes.append(classList[i])
            votedClass.append(max(set(votes), key = votes.count))

        return votedClass
    
    def accuracy(self):
        
        # get classification of datapoints
        votedClasses = self.classifySet(self.testingSet)
        
        # for each datapoint, check if classification is accurate
        targets = self.testingSet[self.target]
        errors = []
        for target, classification in zip(targets, votedClasses):
            if target == classification:
                errors.append(True)
            else:
                errors.append(False)

         # calculate accuracy        
        return np.mean(errors)


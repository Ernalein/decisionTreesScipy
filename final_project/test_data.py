import pandas as pd
import numpy as np
from tree import Node

class test_data:

    '''
    The class test_data tests the accuracy of a decisiontree with the testing set of the data.

    The function classify() is a recursive function and returns the 
    decisiontree's classification of this datapoint.
    The function classifySet() returns the classification of the whole test data.
    The function accuracy() is for calculating the error rate of the decisiontree. 
    The calculated error lies inbetween 0 and 1 where 1 indicates a fully right classification 
    of the testing set and 0 indicaties a fully wrong classification.

    The class only takes a trained decisiontree for testing 
    and the test data has to be a pd.Dataframe.
    '''
    
    def __init__(self, testData:pd.DataFrame, target, node:Node):
        self.testData = testData
        self.target = target
        self.rootNode = node

        # check whether node is trained:
        if node.getAttribute() is None:
            raise TypeError("node has to be part of a trained Decisiontree")

    
    def classify(self, datapoint, node):
        
        # basecase 1: 
        # get the classification of the leaf node
        if node.isLeaf() == True:
            return node.getClassification()


        # recursive case:
        # traverse the tree according to attributes and values:
        
        # get the attribute of the current node and the attribute value of the datapoint
        attribute = node.getAttribute()
        dataValue = datapoint.loc[attribute]

        # itterate through all children of the node
        for child in node.getChildren():
            cValue = child.getValue()
            
            # for interval values
            if child.valueIsContinuous:
                # check whether datapoint lies inbetween the bordervalues
                if dataValue >= cValue[0] and dataValue < cValue[1]:
                    return self.classify(datapoint, child)
            # for discrete values
            elif cValue is dataValue:
                return self.classify(datapoint, child)
        
        # basecase 2:
        # if there are no children with the right value at decision node, get current classification
        return node.getClassification()
        

    def classifySet(self):
        classes = []
        
        # itterate through the test data and classify each datapoint
        for i in range(len(self.testData)):
            datapoint = self.testData.iloc[i]
            classes.append(self.classify(datapoint, self.rootNode))
        return classes
    

    def accuracy(self):
        classes = self.classifySet()
        targets = self.testData[self.target]
        
        errors = []
        # itterates through tagests and classes
        # checks whether the classification of the decisiontree was right or not
        for target, classification in zip(targets, classes):
            if target == classification:
                errors.append(1)
            else:
                errors.append(0)
                
        return np.mean(errors)


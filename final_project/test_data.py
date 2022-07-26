import pandas as pd
import numpy as np
from tree import Node

class test_data:
    
    def __init__(self, testData, target, node:Node):
        self.testData = testData
        self.target = target
        self.rootNode = node

        # check whether node is trained:
        if node.getAttribute() is None:
            raise TypeError("node has to be part of a trained Decisiontree")

    
    def classify(self, datapoint, node):
        
        # get leaf node classification (basecase)
        if node.isLeaf() == True:
            return node.getClassification()

        # traverse down the tree with the decision nodes (recursive case)
        else:
            attribute = node.getAttribute()
            dataValue = datapoint.loc[attribute]
            for child in node.getChildren():
                cValue = child.getValue()
                # for interval values
                if child.valueIsContinuous:
                    if dataValue >= cValue[0] and dataValue < cValue[1]:
                        return self.classify(datapoint, child)
                # for discrete values
                elif cValue is dataValue:
                    return self.classify(datapoint, child)
        
        # if there are no children with the right value at decision node, get current classification (base case)
        return node.getClassification()
        
    def classifySet(self):
        classes = []
        #print("testData: ", self.testData)
        for i in range(len(self.testData)):
            datapoint = self.testData.iloc[i]
            #print("datapoint: ", datapoint)
            classes.append(self.classify(datapoint, self.rootNode))
        return classes
    
    def accuracy(self):
        classes = self.classifySet()
        targets = self.testData[self.target]
        
        errors = []
        for target, classification in zip(targets, classes):
            #print("target: ", target , " class: ", classification)
            if target == classification:
                errors.append(True)
            else:
                errors.append(False)
                
        return np.mean(errors)


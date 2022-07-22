import pandas as pd
from tree import Node

class test_data:
    
    def __init__(self, testData, target, node:Node):
        self.testData = testData
        self.target = target
        self.node = node

        # check whether node is trained:
        if node.getAttribute() is None:
            raise TypeError("node has to be part of a trained Decisiontree")


    def calcError(self, datapoint):

        # compare leaf node classification and datapoint classification (basecase)
        if self.node.isLeaf() == False:
            return self.node.getClassification() == datapoint[self.target]

        # traverse down the tree with the decision nodes (recursive case)
        else:
            attribute = self.node.getAttribute()
            dataValue = datapoint[attribute]
            for child in self.node.children:
                if child.getValue() is dataValue:
                    return test_data(datapoint, self.target, child).calcError()
        
        # if there are no children with the right value at decision node, use current classification (base case)
        return self.node.getClassification() == self.target

    
    def calcClassification(self, datapoint):
        
        # get leaf node classification (basecase)
        if self.node.isLeaf() == False:
            return self.node.getClassification()

        # traverse down the tree with the decision nodes (recursive case)
        else:
            attribute = self.node.getAttribute()
            dataValue = datapoint[attribute]
            for child in self.node.children:
                if child.getValue() is dataValue:
                    return test_data(datapoint, self.target, child).calcClassification()
        
        # if there are no children with the right value at decision node, get current classification (base case)
        return self.node.getClassification()
        
    def classify(self):
        classificationArray = []
        for i in range(self.testData.shape[0]):
            datapoint = self.testData.loc[i]
            classificationArray.append(self.calcClassification(datapoint))
        
        return classificationArray
    
    def accuracy(self):
        errorArray = []
        for i in range(self.testData.shape[0]):
            datapoint = self.testData.loc[i]
            errorArray.append(self.calcError(datapoint))
        
        return np.mean(errorArray)


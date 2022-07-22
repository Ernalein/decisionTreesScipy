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
        # accepts only pd series as datapoint --------------------------------------not sure if this makes sense
        if not isinstance(datapoint, pd.Series):
            raise TypeError("Datapoint has to be of type pd.Series")
        # returns an error array
        error = False

        if self.node.isLeaf():
            classification = self.node.getClassification()
            if classification is not self.target:
                error = True
           
            
        else:

            attribute = self.node.getAttribute()
            value = datapoint[attribute]
            for child in self.node.children:
                targetValue = child.getValue()
                if targetValue is value:
                    self.node = child
                    test_data(datapoint, self.target, child).calcError()

        return error
    
    def classify(self):
        errorArray = []
        for i in range(self.testData.shape[0]):
            datapoint = self.testData.loc[i]
            errorArray.append(self.calcError(datapoint))
        
        return errorArray
    
    


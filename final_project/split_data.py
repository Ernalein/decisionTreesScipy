# this is the split_data file
import numpy as np
import pandas as pd
from tree import Node
 
class split_data:
    def __init__(self, data, target, attributes, node = Node):
        self.data = data
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data has to be a Pandas Dataframe")
        
        self.target = target
        if not isinstance(self.target, str):
            raise TypeError("Taget has to be of type string")
        
        self.attributes = attributes
        if not isinstance(attributes, list):
            raise TypeError("Attributes have to have structure list")
        for attribute in self.attributes:
            if not isinstance(attribute, str):
                raise TypeError("Attributes have to be of type string")

        self.node = node
        
    def entropy(self):
        targetColumn = self.data.loc[:, self.target]

        values = set(targetColumn)
        entropySum = 0
        for value in values:
            p = targetColumn.count(value)/ len(targetColumn)
            entropySum = entropySum + (- p * np.log(p))

        return entropySum
    

    def informationGain(self, attribute):
        attributeColumn = self.data.loc[:, attribute]
        values = set(attributeColumn)
        gainSum = 0
        
        for value in values:
            subsetData = self.data[attributeColumn == value]
            subset = split_data(subsetData, self.target)
            # claculate entropy and normalize by size of subsets
            gainSum = gainSum + (subsetData.shape[0] / self.data.shape[0]) * subset.entropy()

        # substract summed and weighted entropy of subsets from entropy of whole set    
        infoGain = self.entropy() - gainSum

        return infoGain


    def chooseAttribute(self):
        maxGain= 0
        maxAttribute = ""

        # calculate Information Gain for each attribute
        for attribute in self.attributes:

            gain = self.informationGain(attribute)
            if gain > maxGain:
                maxGain = gain
                maxAttribute = attribute

        # choose attribute with highest Information Gain
        return maxAttribute

    def classify(self):
        targetColumn = self.data.loc[:, self.target]
        values = set(targetColumn)
        maxClass = 0
        classification = ""
        for value in values:
            p = targetColumn.count(value)/ len(targetColumn)
            if p > maxClass:
                maxClass = p
                classification = value

        return classification


    def id3(self):
        # base cases:
        # 1) all instances have same target value -> leaf node with target value
        if (self.data[self.target].nunique() is 1):
            self.node.setClassification(self.data[self.target][0])
            return 
        # 2) out of discriptive features -> leaf node with majority of target values
        if (not self.attributes):
            self.node.setClassification(self.classify())
            return
        # 3) no instances left in dataset -> take majority of parent node
        if (self.data is None):
            parent = self.node.getParent()
            self.node.setClassification(parent.getClassification())
            return

        # recursive case:
        # choose attribute with highest explainatory power
        attribute = self.chooseAttribute()
        self.node.setAttribute(attribute)
        self.node.setClassification(self.classify())

        # split data according to attribute
        attributeColumn = self.data.loc[:, attribute]
        values = set(attributeColumn)
        for value in values:
            subsetData = self.data[attributeColumn == value]
            childNode = Node(parent=self.node, value=value)
            self.node.setChild(childNode)
            subset = split_data(subsetData, self.target, self.attributes.remove(attribute), childNode)
            # recursive call on all partitions
            subset.id3()
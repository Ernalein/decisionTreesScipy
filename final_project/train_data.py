import numpy as np
import pandas as pd
from tree import Node

class train_data:
    '''
    class train_data has all important functions for ID3 Algorithm:
    it can calculate the entropy of some data, the information gain, choose an attriute.
    within the ID3 algorithm a tree will be trained.
    the function retrain(data) retrains a trained tree with new data.
    '''
    def __init__(self, data, target, attributes, node:Node = None, recursion_depth = None, continuous_splitting = 0.1,  max_recursion = 10):
        
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
        self.continuous_splitting = continuous_splitting
        self.recursion_depth = recursion_depth
        self.max_recursion = max_recursion
    
    def is_continuous(self, values):
        # checks is variable is a continuous variable
        if len(values) > 10:
            for value in values:
                if value is not int or float:
                    return False
            return True
        else:
            return False

    def attribute_continuous(self, attributeColumn, values):
        
        # sort the set
        values = list(sorted(values))
        split_lenght = int(len(values) * self.continuous_splitting)
        maxGain = 0
        maxValue = 0
        for split in range(split_lenght):
            value = values[split]

            subsetData1 = self.data[attributeColumn <= value]
            subsetData2 = self.data[attributeColumn > value]
            subset1 = train_data(subsetData1, self.target, self.attributes)
            subset2 = train_data(subsetData2, self.target, self.attributes)

            gainSum =  (subsetData1.shape[0] / self.data.shape[0]) * subset1.entropy() + (subsetData2.shape[0] / self.data.shape[0]) * subset2.entropy()
            infoGain = self.data.entropy() - gainSum

            if infoGain > maxGain:
                maxGain = infoGain
                maxValue = value
        
        return [maxGain, maxValue]

    def entropy(self):
        targetColumn = self.data.loc[:, self.target]

        values = set(targetColumn)
        entropySum = 0
        for value in values:
            # p = targetColumn.count(value) / len(targetColumn)
            valueColumn = self.data[targetColumn == value].loc[:, self.target]
            p = len(valueColumn)/ len(targetColumn)
            entropySum = entropySum + (- p * np.log(p))

        return entropySum
    

    def informationGain(self, attributeColumn, values):
        gainSum = 0
        
        for value in values:
            
            subsetData = self.data[attributeColumn == value]
            subset = train_data(subsetData, self.target, self.attributes)
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
            attributeColumn = self.data.loc[:, attribute]
            values = set(attributeColumn)
            gain = 0

            # calculate Information gain for this attribute
            if self.is_continuous(values):
                gain = self.attribute_continuous(attributeColumn, values)[0] # calculates the split information ???
            else:
                gain = self.informationGain(attributeColumn, values)

            # store attribute with highest information gain
            if gain >= maxGain:
                maxGain = gain
                maxAttribute = attribute

        # choose attribute with highest Information Gain
        return maxAttribute

    def classify(self):
        # returns the most commen classification of the dataset
        
        targetColumn = self.data.loc[:, self.target]
        values = set(targetColumn)
        maxClass = 0
        classification = ""
        for value in values:
            
            # if targetColumn.count(value) > maxClass:
            valueColumn = self.data[targetColumn == value].loc[:, self.target]
            if len(valueColumn) > maxClass:
                maxClass = len(valueColumn)
                classification = value

        return classification


    def remove_attribute(self, attribute):
        self.attributes.remove(attribute)
        return self.attributes

    def id3(self):
        # base cases:
        # 1) all instances have same target value -> leaf node with target value
        if (self.data[self.target].nunique() == 1):
            self.node.setClassification(self.data[self.target].iloc[0])
            print("basecase1")
            return 
        # 2) out of discriptive features -> leaf node with majority of target values
        if (not self.attributes):
            self.node.setClassification(self.classify())
            print("basecase2")
            return
        # 3) no instances left in dataset -> take majority of parent node
        if (self.data is None):
            parent = self.node.getParent()
            self.node.setClassification(parent.getClassification())
            print("basecase3")
            return
        # 4) maximal recursion depth:
        if self.recursion_depth == self.max_recursion:
            self.node.setClassification(self.classify())
            print("basecase4")
            return


        # recursive case:
        # choose attribute with highest explainatory power
        print("in recursion")
        print("attributs: ", self.attributes)
        attribute = self.chooseAttribute()
        self.node.setAttribute(attribute)
        self.node.setClassification(self.classify())

        # split data according to attribute
        attributeColumn = self.data.loc[:, attribute]
        values = set(attributeColumn)
        new_attributes = self.remove_attribute(attribute)
        recursion_depth = self.recursion_depth + 1

        # chosen attribute is a continuous variable:
        if self.is_continuous(values):
            print("continuous")
            value = self.attribute_continuous(attributeColumn, values)[1]

            subsetData1 = self.data[attributeColumn <= value]
            subsetData2 = self.data[attributeColumn > value]
            childNode1 = Node(parent=self.node, value=f"<= {value}")
            childNode2 = Node(parent=self.node, value=f"> {value}")
            self.node.setChild(childNode1)
            self.node.setChild(childNode2)
            
            subset1 = train_data(data=subsetData1, target=self.target, attributes=new_attributes, node=childNode1, recursion_depth=recursion_depth)
            subset2 = train_data(data=subsetData2, target=self.target, attributes=new_attributes, node=childNode2, recursion_depth=recursion_depth)
            # recursive call on all partitions
            subset1.id3()
            subset2.id3()
        
        # chosen attribute is a categorical variable:
        else:
            print("not continuous")
            for value in values:
                subsetData = self.data[attributeColumn == value]
                childNode = Node(parent=self.node, value=value, target=self.target)
                self.node.setChild(childNode)
                subset = train_data(data=subsetData, target=self.target, attributes=new_attributes, node=childNode, recursion_depth=recursion_depth)
    
                # recursive call on all partitions
                subset.id3()

    def retrain(self, data):
        self.data = data
        self.id3()
    
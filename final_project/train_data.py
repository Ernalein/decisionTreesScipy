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
    
    ######################################
    ## methods for continuous variables ##
    ######################################
    
    def is_continuous(self, values):
        # checks is variable is a continuous variable
        # (it is continuous if it has more than 10 different values and is a numericla scalar)
        if len(values) > 10:
            if isinstance(list(values)[5], int) or isinstance(list(values)[5], float):
                return True
        return False

    
    def getBoundaries(self, tColumn, aColumn):
        # by looking at the target column and the attribute column the
        # function decides on decision boundaries in a continuous varibale, where classification changes
        # aColumn -> attribute column with the continuous values
        # tColumn -> target column with the classification
        
        # 1) sort the two columns by attribute values
        columns = pd.DataFrame(data={"a":list(aColumn), "t":list(tColumn)}).sort_values(by="a")
        columns.index = range(len(columns))

        # 2) find decision boundaries where classification changes
        leftBound = np.NINF # first interval has negative infinity as left boundary
        rightBound = None
        boundaries = []
        currentClass = columns["t"][0]
        
        for i in range(len(columns)):
            
            # when classification changes
            if(columns["t"][i] != currentClass):
                currentClass = columns["t"][i]
                
                # get the value in the middel of the values where classification changes
                beforeSwitch = columns["a"][i-1]
                afterSwitch = columns["a"][i]
                rightBound = (beforeSwitch + afterSwitch) / 2

                # safe the tupple of two boundaries 
                # represents an interval with a uniform classification
                boundaries.append((leftBound, rightBound))
                leftBound = rightBound
        
        # last interval has negative infinity as right boundary
        boundaries.append((leftBound, np.PINF))
        
        # if the getBoundaries function returns more then 10 intervals
        # set intervals indipendent of classification
        if len(boundaries) > 10:
            return self.setBoundaries(aColumn)
        
        return boundaries
    
    def setBoundaries(self, aColumn):
        # if the getBoundaries function returns more then 10 intervals
        # sets 10 eaqually sized intervals indipendent of classification
        
        # calculate size of intervals
        maximum = np.max(aColumn)
        minimum = np.min(aColumn)
        stepsize = (maximum - minimum)/ 10
        boundaries = []
        
        # make a tupel for each interval
        leftBound = np.NINF
        rightBound = minimum + stepsize
        for i in range(9):
            boundaries.append((leftBound, rightBound))
            leftBound = rightBound
            rightBound = leftBound + stepsize
        boundaries.append((leftBound, np.PINF))
        
        return boundaries
        
    
    def replaceContinuous(self, boundaries, aColumn):
        # replaces the continuous values of an attribute by the
        # tuples that represent an interval
        
        newAColumn = []
        for value in aColumn:
            for l, r in boundaries:
                if value >= l and value < r:
                    newAColumn.append((l, r))
        return newAColumn
    
    
    #######################################
    ## methods for choosing an attribute ##
    #######################################
    
    
    def entropy(self):
        # calculates entropy 
        targetColumn = self.data.loc[:, self.target]

        values = set(targetColumn)
        entropySum = 0
        for value in values:
            p = list(targetColumn).count(value) / len(targetColumn)
            entropySum = entropySum + (- p * np.log(p))

        return entropySum
    

    def informationGain(self, attributeColumn, values):
        # calculates the informationGain
        
        gainSum = 0
        for value in values:
            
            mask = lambda aColumn, value :(row == value for row in aColumn) 
            subsetData = self.data.iloc[mask(attributeColumn, value),:]
            subset = train_data(subsetData, self.target, self.attributes)
            # claculate entropy and normalize by size of subsets
            gainSum = gainSum + (subsetData.shape[0] / self.data.shape[0]) * subset.entropy()

        # substract summed and weighted entropy of subsets from entropy of whole set    
        infoGain = self.entropy() - gainSum

        return infoGain

    def gainRatio(self, attributeColumn, values):
        # calculating the Gain Ratio instead of the InforamtionGain
        # to prefer attributes with few values
        
        infoGain = self.informationGain(attributeColumn, values)

        splitInfo = 0.0
        for value in values:
            
            subset = attributeColumn[attributeColumn == value]
            # proportion of subset size and whole set size
            s = len(subset) / len(attributeColumn)
            if s != 0.0:
                splitInfo = splitInfo + ((- s) * np.log(s))
        
        # to avoid dividing by zero
        if splitInfo == 0:
            splitInfo = infoGain
            if infoGain == 0:
                return 0
            
        return infoGain / splitInfo
        

    def chooseAttribute(self):
        # chooses an attribute that maximises GainRatio
        #   
        maxGain= 0
        maxAttribute = ""

        # calculate Gain Ratio for each attribute
        for attribute in self.attributes:
            
            attributeColumn = self.data.loc[:, attribute]
            values = set(attributeColumn)
            gain = 0

            # replace the values in attributeColumn with continuous values by Intervals
            if self.is_continuous(values):
                targetColumn = self.data[self.target]
                boundaries = self.getBoundaries(targetColumn, attributeColumn)
                attributeColumn = self.replaceContinuous(boundaries, attributeColumn)
                values = set(attributeColumn)            
                
            # calculate gainRatio
            gain = self.gainRatio(attributeColumn, values)

            # store attribute with highest information gain
            if gain >= maxGain:
                maxGain = gain
                maxAttribute = attribute

        # choose attribute with highest Information Gain
        return maxAttribute

    ############################################
    ## methods for building the decision tree ##
    ############################################
    
    def classify(self):
        # returns the most commen classification of the dataset
        
        targetColumn = self.data.loc[:, self.target]
        values = set(targetColumn)
        maxClass = 0  # highest number of values
        classification = "" # classification of most common value
        for value in values:
            # check if calssification value is more common then other classification values
            if list(targetColumn).count(value) > maxClass:
                maxClass = list(targetColumn).count(value)
                classification = value

        return classification

    def id3(self):
        # base cases:
        # 1) all instances have same target value -> leaf node with target value
        if (self.data[self.target].nunique() == 1):
            self.node.setClassification(self.data[self.target].iloc[0])
            print("basecase1")
            print("classification: ", self.data[self.target].iloc[0])
            return 
        # 2) out of discriptive features -> leaf node with majority of target values
        if (not self.attributes):
            self.node.setClassification(self.classify())
            print("basecase2")
            print("classification: ", self.classify())
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
        new_attributes = self.attributes
        new_attributes.remove(attribute)
        recursion_depth = self.recursion_depth + 1

        valueIsContinuous=False
        
        # chosen attribute is a continuous variable:
        if self.is_continuous(values):
            print("continuous")
            
            targetColumn = self.data[self.target]
            boundaries = self.getBoundaries(targetColumn, attributeColumn)
            attributeColumn = self.replaceContinuous(boundaries, attributeColumn)
            values = set(attributeColumn)
            valueIsContinuous=True
        
        # create leaf node for each attribute value
        for value in values:
            # get the subset determined by the attribute value
            mask = lambda aColumn, value :(row == value for row in aColumn) 
            subsetData = self.data.iloc[mask(attributeColumn, value),:]
            # create a node in the tree
            childNode = Node(parent=self.node, value=value, valueIsContinuous=valueIsContinuous, target=self.target)
            self.node.setChild(childNode)
            # train the node with the data subset
            subset = train_data(data=subsetData, target=self.target, attributes=new_attributes, node=childNode, recursion_depth=recursion_depth)
            subset.id3() # recursive call on all partitions
            

    def retrain(self, data):
        self.data = data
        self.id3()
    
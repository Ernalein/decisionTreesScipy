import numpy as np

class Node:
    '''
    The class Node has all the getter and setter functions that are necessary for a tree. 

    Furthermore it has additional setter and getter for the attributes, values and classification
    and a function for printing the tree. It also has a variable to indicate whether or not the 
    attribute had continous variables.

    The decision tree knows its attributes and the corresponding values in the following way:
    The parent node knows the attribute according to which the data was split.
    The children know the values they take according to the attribute of the parent node.
    '''
    
    def __init__(self, parent = None, attribute = None, classification = None, value = None, valueIsContinuous = False, target = None):
        self.children = []  
        self.parent = parent
        self.attribute = attribute
        self.classification = classification
        self.value = value
        self.valueIsContinuous = valueIsContinuous
        self.target = target

    
    # typical setter and getter for tree structure 
    def setChild(self, node):
        self.children.append(node)
        
    def setParent(self, node):
        if (self.parent is not None):
            self.parent.children.remove(self)
        self.parent = node

    def getChildren(self):
        return self.children
    
    def getParent(self):
        return self.parent
    

    # functions to check whether node is a leaf or root node
    def isLeaf(self):
        if len(self.children) > 0:
            return False
        else:
            return True

    def isRoot(self):
        if (self.parent is None):
            return True
        else:
            return False
    

    # -------------------------------------------------------------------------------------------
    # setter and getter for attributes, values and classification 

    def setAttribute(self, attribute):
        self.attribute = attribute
    
    def getAttribute(self):
        return self.attribute
    
    def setClassification(self, classification):
        self.classification = classification
    
    def getClassification(self):
        return self.classification
    
    def setValue(self,value):
        self.value = value

    def getValue(self):
        return self.value
    
    #-------------------------------------------------------------------------------------------
    # function to print the tree
    def printTree(self, level = 0):
        tab = "    "
        if level == 0:
            print("Decision tree:")
        if self.isLeaf():
            print(tab*level, "classification: ", self.target, " = " , self.classification)
        else:
            for child in self.children:
                # printing intervals
                if child.valueIsContinuous:
                    interval = "" 
                    if child.value[0] == np.NINF:
                        interval = f"smaller then {child.value[1]}"
                    elif child.value[1] == np.PINF:
                        interval = f"bigger then {child.value[0]}"
                    else:
                        interval = f"between {child.value[0]} and {child.value[1]}"
                    print(tab*level, self.attribute, ": ", interval)
                
                # printing discrete values
                else:
                    print(tab*level, self.attribute, ": ", child.value)
                
                # traverse deeper into the tree
                child.printTree(level + 1)
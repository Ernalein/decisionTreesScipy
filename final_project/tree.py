class Node:
    '''
    class Node innitialises a tree structure for a non-binary tree
    it has the typical setter and getter methods and a method to remove a 
    child from the list of children
    '''
    def __init__(self, parent = None, attribute = None, classification = None, value = None, target = None):
        self.children = []  
        self.parent = parent
        self.attribute = attribute
        self.classification = classification
        self.value = value
        self.target = target
        

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
    
    def deleteChild(self, node):
        if (node in self.children):
            node.parent = None
            self.children.remove(node)
        else:
            raise TypeError("Child not in Children")
        
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
    

    def printTree(self, level = 0):
        tab = "    "
        if self.isLeaf():
            print(tab*level, "classification: ", self.target, " = " , self.classification)
        else:
            for child in self.children:
                print(tab*level, self.attribute, ": ", child.value)
                child.printTree(level + 1)
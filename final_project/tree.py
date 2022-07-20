from attr import attributes


class Node:
    '''
    class Node innitialises a tree structure for a non-binary tree
    it has the typical setter and getter methods and a method to remove a 
    child from the list of children
    '''
    def __init__(self, children = [], parent = None, attribute = None, classification = None, value = None):
        self.children = children
        self.parent = parent
        self.attribute = attribute
        self.classification = classification
        self.value = value

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
        if (not self.children):
            return True
        else:
            return False

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
        if self.isLeaf() is True:
            print(f"{tab*level}classification: {self.classification}")
            return

        print(f"{tab*level}{self.attribute}: {self.value}")
        level = level + 1
        for child in self.children:
            
            child.printTree(level)
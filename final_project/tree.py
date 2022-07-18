class Node:
    def __init__(self, data, left = None, right = None, parent = None):
        self.left = left
        self.right = right
        self.parent = parent
        self.data = data

    def setLeft(self, Node):
        if (self.left is not None):        
            self.left.parent = None
        self.left = Node
        
    def setRight(self, Node):
        if (self.right is not None):
            self.right.parent = None
        self.right = Node

    def setParent(self, Node):
        if (self.parent is not None):
            if (Node == self.parent.left):
                self.parent.left = None
            else:
                self.parent.right = None
        self.parent = Node
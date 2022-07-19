import numpy as np
import pandas as pd
from tree import Node
from split_data import split_data
from prepare_data import prepare_data
# this is the main

# 1. load data
# data = ....

# 2. prepare data
data = prepare_data(data)

# split_data with id3
rootNode = Node()
decisionTree = split_data(data, target, attributes, rootNode)
decisionTree.id3()

rootNode.printTree()

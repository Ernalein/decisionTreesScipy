import numpy as np
import pandas as pd
from tree import Node
from train_data import train_data
from prepare_data import prepare_data
from test_data import test_data
# this is the main

# 1. load data
# data = ....

# 2. prepare data
data = prepare_data(data)

# choose the target value 
target = "some string" # to be choosen
attributes = list(data[0][0].columns).remove(target)

# 3. split_data with id3
rootNode = Node()
decisionTree = train_data(data[1][0], target, attributes, rootNode)
decisionTree.id3()

rootNode.printTree()

# calculate error of test set:
error = test_data(data[0][0], target, rootNode).classify()



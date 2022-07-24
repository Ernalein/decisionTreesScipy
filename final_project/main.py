import pandas as pd
from tree import Node
from train_data import train_data
from prepare_data import prepare_data
from test_data import test_data
# this is the main

# 1. load data
d= {"gender": ["f", "f", "f", "f", "f", "m", "m", "m", "m", "m", "f", "f", "f", "f", "m", "f", "m", "m", "m", "m"],
    "vegan": [True, True, True, False, False, True, False, False, False, False, True, True, True, False, False, True, False, True, False, True],
    "coxi": [True, True, True, False, True, True, True, False, False, False, True, False, False, True, False, False, True, True, True, False],
    "green": [True, True, True, False, False, True, False, True, False, True, True, True, True, False, True, True, True, False, False, False],
    "party_lover": [True, False, False, True, False, False, True, True, True, False, True, True, True, False, True, True, True, False, False, False],
    "abschluss": ["Bachelor", "Bachelor", "Master", "Keiner", "Keiner", "Master", "Master", "PhD", "Keiner", "PhD", "Master", "Master", "PhD", "Keiner", "PhD", "Bachelor", "Master", "Keiner", "Keiner", "Master"]}

data = pd.DataFrame(data = d)


# 2. prepare data
prepared_data = prepare_data(data)



# choose the target value 
target = "vegan"
attributes = list(data.columns)
attributes.remove(target)
error = []
# 3. split_data with id3
for chunk in range(10):
    testing_data = prepared_data[chunk][0]
    training_data = prepared_data[chunk][1]
    rootNode = Node(root=True, children=[])
    decisionTree = train_data(data=training_data, target=target, attributes=attributes, node=rootNode, recursion_depth=0)
    decisionTree.id3()

    print('')
    print("Tree ", chunk)
    print('')

    rootNode.printTree()

    # calculate error of test set:
    test_error = test_data(testing_data, target, rootNode).accuracy()
    print("appendet error: ", test_error)
    error.append(test_error) 

    print("--------------------------------------------------------")

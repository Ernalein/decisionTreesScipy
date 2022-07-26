import pandas as pd
import seaborn as sns
from tree import Node
from train_data import train_data
from prepare_data import prepare_data
from test_data import test_data
# this is the main




# datasets from seaborn:
# ['anagrams', 'anscombe', 'attention', 'brain_networks', 'car_crashes',
#'diamonds', 'dots', 'exercise', 'flights', 'fmri', 'gammas', 'geyser',
#'iris', 'mpg', 'penguins', 'planets', 'taxis', 'tips', 'titanic']

# check out available dataset form seaborn:
data = sns.load_dataset("titanic")
data = data.dropna(how = "any")
print(data.columns)
print(data)





### build some penguin trees:

data = sns.load_dataset("penguins")

# 2. prepare data
data = prepare_data(data)

# 3. choose the target value
target = "island"

# 4. train a tree for each chunk of the training set
decisionTrees = []
for trainingSet in data[0]:
    
    attributes = list(trainingSet.columns)
    attributes.remove(target)
    
    rootNode = Node()
    decisionTree = train_data(data=trainingSet, target=target, attributes=attributes, node=rootNode, max_recursion = 10)
    decisionTree.id3()
    decisionTrees.append(rootNode)
    rootNode.printTree()

for testingSet, tree in zip(data[1], decisionTrees):
    
    testData = test_data(testingSet, target, tree)
    print(testData.accuracy())

    
    
    
### build some titanic trees

data = sns.load_dataset("titanic")
data = data.drop("alive", axis=1)

# 2. prepare data
data = prepare_data(data)

# 3. choose the target value
target = "survived"

# 4. train a tree for each chunk of the training set
decisionTrees = []
for trainingSet in data[0]:
    
    attributes = list(trainingSet.columns)
    attributes.remove(target)
    
    rootNode = Node()
    decisionTree = train_data(data=trainingSet, target=target, attributes=attributes, node=rootNode, max_recursion = 10)
    decisionTree.id3()
    decisionTrees.append(rootNode)
    rootNode.printTree()

for testingSet, tree in zip(data[1], decisionTrees):
    
    testData = test_data(testingSet, target, tree)
    print(testData.accuracy())
    
    
    
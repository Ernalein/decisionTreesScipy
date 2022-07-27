import pandas as pd
import seaborn as sns
import numpy as np
from Node import Node
from Train_data import Train_data
from prepare_data import prepare_data
from Test_data import Test_data
from Random_forest import Random_forest


# datasets and the corresponding target features for training the decision trees
seabornToyData = [("penguins", "species"), ("titanic", "survived"), ("iris", "species"), ("tips", "sex")]

for dataset,target in seabornToyData:

    # 1. load data
    data = sns.load_dataset(dataset)
    if dataset == "titanic":
        data = data.drop("alive", axis=1)

    # 2. prepare training and testing chunks
    trainingSets, testingSets = prepare_data(data, 0.1)

    # 3. train a tree for each chunk of the training set -> 10 trees
    decisionTrees = []
    for trainingSet in trainingSets:

        attributes = list(trainingSet.columns)
        attributes.remove(target)

        rootNode = Node()
        decisionTree = Train_data(data=trainingSet, target=target, attributes=attributes, node=rootNode, max_recursion = np.PINF)
        decisionTree.id3()
        decisionTrees.append(rootNode)

    # 4. test each tree with the corresponding testing chunk
    accuracies = []
    for testingSet, tree in zip(testingSets, decisionTrees):

        testData = Test_data(testingSet, target, tree)
        accuracies.append(testData.accuracy())

    # 5. print tree with best accuracy score
    maxAcc = np.max(accuracies)
    bestTree = decisionTrees[np.argmax(accuracies)]
    print("\n\nDecision tree trained on ", dataset ," Dataset for classifying ", target)
    print("accuracy score: ", maxAcc)
    print(bestTree.printTree())
    
    # 6. build a decision Forest
    forest = Random_forest(data, target, trainRatio = 0.1, nrTrees = 15, testRatio = 0.1)
    forest.train()
    fAcc = forest.accuracy()
    
    # 7. check weather decision Forest is more accurate
    print("A random forest trained on the same dataset reaches an accuracy of ", fAcc)
    if fAcc > maxAcc:
        print("The random forest performs better then the best decision Tree")
    elif fAcc < maxAcc:
        print("The random forest performs worse then the best decision Tree")
    else:
        print("The random forest performs equal to the best decision Tree")

        
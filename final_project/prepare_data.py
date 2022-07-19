import numpy as np
import pandas as pd
# this is the prepare_data file
# get chunks for training and testing

def prepare_data(data = pd.DataFrame, tratio = 0.1, validation = False):
    # get length of dataframe
    dataLength = data.shape[0]

    # get chunck size:
    chunkSize = int(dataLength * tratio) 

    dataList = []

    for i in range(chunkSize-1):
        dataList.append(data.iloc[i*chunkSize:(i+1)*chunkSize, :])
    
    testList = []
    trainList = []
    if validation is True:
        validationList = []
        for i in range(len(dataList)):
            testList.append(dataList[i])
            if i == len(dataList):
                validationList.append(dataList[0])
            else:
                validationList.append(dataList[i+1])

            helper = dataList[:i] + dataList[i+1 :]
            for j in range(len(helper-1)):
                trainingset = helper[j].append(helper[j+1])
            trainList = trainingset
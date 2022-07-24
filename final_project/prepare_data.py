import pandas as pd
# this is the prepare_data file
# get chunks for training and testing

def prepare_data(data:pd.DataFrame, tratio = 0.1):
    # variables to return
    testData = []
    trainingData = []

    # check whether tratio is smaller than 1
    if tratio >= 1:
        raise TypeError("tratio has to be smaller than 1")
    
    # get length of dataframe
    dataLength = data.shape[0]
    # get chunck size:
    chunkSize = int(dataLength * tratio) 
    # get number of chunks
    nr_chunks = int(1/tratio)

    # append data set to existing dataset
    doubleData = data.append(data)

    # itterate through doubleData, assigh a certain chunk to testing, training and if wished 
    # validation

    for chunk in range(nr_chunks-1):
        testData.append(doubleData.iloc[chunk: chunkSize*(chunk+1), :])
        trainingData.append(doubleData.iloc[chunkSize*(chunk+1): chunkSize*(nr_chunks+chunk), :])
    
    return [testData, trainingData]
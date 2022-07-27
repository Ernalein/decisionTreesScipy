import pandas as pd

def prepare_data(data:pd.DataFrame, tratio = 0.1):
    
    # remove any Nans from Dataframe
    data = data.dropna(how='any')
    
    # shuffle data
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    
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

    # itterate through doubleData, assigh a certain chunk to testing and training
    for chunk in range(nr_chunks-1):
        testData.append(doubleData.iloc[chunkSize*chunk: chunkSize*(chunk+1), :])
        trainingData.append(doubleData.iloc[chunkSize*(chunk+1): chunkSize*(nr_chunks+chunk), :])
    
    return [testData, trainingData]
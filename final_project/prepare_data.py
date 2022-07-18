import numpy as np
import pandas as pd
# this is the prepare_data file
# get chunks for training and testing

class prepare_data(data = pd.DataFrame):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.feature = list(data.column).remove(target)

    # split the data according to percentage
    
    
    

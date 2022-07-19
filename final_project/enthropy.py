
import numpy as np
import pandas as pd
from . import tree

def entrophy(parent = Node):
    # take children and their proposition of target values?
    # calculate sum over children(-pi*log2(pi))
    parent_data = parent.data
    left_data = parent.left.data
    right_data = parent.right.data

    # calculate proportion of target values in parent
    



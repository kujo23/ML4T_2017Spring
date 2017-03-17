"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.mgrid[-5:5:0.5,-5:5:0.5].reshape(2,-1).T
    Y = X[:,0]*X[:,1] + np.random.normal(size = X.shape[0])
    return X, Y

def best4RT(seed=1489683273):
    np.random.seed(seed)
    X = np.random.normal(size = (50, 2)) 
    Y = 0.8 * X[:,0] + 5.0 * X[:,1] 
    return X, Y

if __name__=="__main__":
    print "they call me Tim."

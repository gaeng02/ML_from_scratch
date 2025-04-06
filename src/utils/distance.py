import numpy as np

# Calculate Distance between two vectors

def Euclidean (a, b) :
    '''
    Returns :
        float 
    '''

    return np.sqrt(np.sum((a - b) ** 2))

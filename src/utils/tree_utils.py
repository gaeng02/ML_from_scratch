import numpy as np

def entropy (y) :
    '''
    Calculate the entropy

    Args :
        y (np.ndarray) : label array

    Returns :
        entropy (float) : entropy value (>= 0)

    '''

    labels, counts = np.unique(y, return_counts = True)
    probabilities = counts / counts.sum()

    entropy_sum = 0
    
    for p in probabilities :
        if (p > 0) :
            entropy_sum += p * np.log2(p)

    entropy = - entropy_sum

    return entropy

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

def gini_index (y) : 
    '''
    Calculate the Gini index

    Args :
        y (np.ndarray) : label array

    Returns :
       gini (float) : gini index (0 ~ 1)
    '''

    labels, counts = np.unique(y, return_counts = True)
    probabilities = counts / counts.sum()

    gini = 1 - np.sum(probabilities ** 2)

    return gini

def information_gain (y, left, right, criterion = "entropy") :
    '''
    Calculate the information gain as a potential split

    Args :
        y (np.ndarray) : parent node
        left (np.ndarray) : left child labels
        right (np.ndarray) : right child labels
        criterion (str) : "entropy" or "gini_index" (default = "entropy")

    Returns :
        float : information gain
    '''

    if (criterion == "entropy") : measure = entropy
    elif (criterion == "gini") : measure = gini_index
    else :
        raise ValueError("")

    n = len(y)
    n_left = len(left)
    n_right = len(right)

    gain = measure(y) - (n_left / n) * measure(left) - (n_right / n) * measure(right)
    
    return gain

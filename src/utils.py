import numpy as np

def train_test_split (X, y, test_size = 0.2, shuffle = True, seed = None) :
    '''
    Split the dataset into training and testing dataset
    
    Args :
        X (np.ndarray) : feature data
        y (np.ndarray) : label data
        test_size (float) : proportion of test dataset (default = 0.2)
        shuffle (bool) : shuffling or not (default = True)
        seed (int or None) : random seed for shuffling (default = None)

    Returns :
        X_train (np.ndarray)
        X_test (np.ndarray)
        y_train (np.ndarray)
        y_test (np.ndarray)

    ''' 

    if shuffle :
        
        if seed is not None :
            np.random.seed(seed)
            
        size = X.shape[0]
        indices = np.arange(size)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
            
    split = int(size * (1 - test_size))
    
    return X[ : split], X[split : ], y[ : split], y[split : ]

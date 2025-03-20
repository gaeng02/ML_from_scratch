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


def accuracy_score (y_true, y_pred) :
    '''
    Calculate the accuracy score 
    
    Args :
        y_true (np.ndarray) : truth binary label
        y_pred (np.ndarray) : predicted binary label

    Returns :
        float (0 ~ 1) : accuracy
    '''
    
    return np.mean(y_true == y_pred)

# TP : True Positive. predict 1, actually 1
# FP : False Positive. predict 1, actually 0
# TN : True Negative. predict 0, actually 0
# FN : False Negative. predict 0, actually 1

def precision_score (y_true, y_pred) :
    '''
    Calculate the precision score 
    
    Args :
        y_true (np.ndarray) : truth binary label
        y_pred (np.ndarray) : predicted binary label

    Returns :
        float (0 ~ 1) : precision
    '''

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    if (TP == 0) : return 0.0
    if (FP == 0) : return 1.0
    
    return TP / (TP + FP)


def recall_score (y_true, y_pred) :
    '''
    Calculate the recall score 
    
    Args :
        y_true (np.ndarray) : truth binary label
        y_pred (np.ndarray) : predicted binary label

    Returns :
        float (0 ~ 1) : recall
    '''

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if (TP == 0) : return 0.0
    if (FN == 0) : return 1.0
    
    return TP / (TP + FN)

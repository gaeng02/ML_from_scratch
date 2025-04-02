import numpy as np
import matplotlib.pyplot as plt

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


def f1_score (y_true, y_pred) :
    '''
    Calculate the f1 score 
    
    Args :
        y_true (np.ndarray) : truth binary label
        y_pred (np.ndarray) : predicted binary label

    Returns :
        float (0 ~ 1) : f1 score 
    '''
    
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    if (p + r == 0) : return 0

    return 2 * p * r / (p + r)


def confusion_matrix (y_true, y_pred) :
    '''
    Draw the confusion matrix  
    
    Args :
        y_true (np.ndarray) : truth binary label
        y_pred (np.ndarray) : predicted binary label

    '''

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    print(" === Confusion Matrix === ")
    print()
    print("                 Prediction")
    print("               |   0   |   1   |")
    print("         ---------------------")
    print(f" Actual   0   | {TN : 3}   | {FP : 3}   |")
    print(f"           1   | {FN : 3}   | {TP : 3}   |")


def roc_curve (y_true, y_score, thresholds = 100) :
    '''
    Calculate FPR and TPR for ROC Curve
    
    Args :
        y_true (np.ndarray) : True binary labels
        y_score (np.ndarray) : Predicted scores or probabilities
        thresholds (int) : Number of thresholds to evaluate (default : 100)
    
    Returns :
        FPR, TPR (list of float) : fpr and tpr
    '''
    
    thresholds = np.linspace(0, 1, thresholds)
    FPR = []
    TPR = []

    for thresh in thresholds :
        y_pred = (y_score >= thresh).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        if ((FP + TN) == 0) : fpr = 0
        else : fpr = FP / (FP + TN)

        if ((TP + FN) == 0) : tpr = 0
        else : fpr = TP / (TP + FN)

        FPR.append(fpr)
        TPR.append(tpr)

    return FPR, TPR


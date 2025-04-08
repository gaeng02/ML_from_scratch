import numpy as np
from src.distance import Euclidean

class KNNClassifier :

    def __init__ (self, k = 3) :
        self.k = k
    
    def fit (self, X, y) :
        self.X_train = X
        self.y_train = y

    

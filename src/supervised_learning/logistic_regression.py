import numpy as np
from src.utils import Sigmoid

class LogisticRegression () :
    
    def __init__ (self, learning_rate = 0.001, epochs = 500) : 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sigmoid = Sigmoid()
        
    def _initialize (self, n_features) :
        self.weights = np.zeros(n_features)
        self.bias = 0.0 
    
    def fit (self, X, y) :
        '''
        Args :
            X (np.ndarray) : X_train
            y (np.ndarray) : y_train
        
        '''
        n_samples, n_features = X.shape
        self._initialize(n_features)

        for epoch in range (self.epochs) :
            
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            self.weights -= self.learning_rate * dw

            db = (1 / n_samples) * np.sum(error)
            self.bias -= self.learning_rate * db
            
    def predict_proba (self, X) :
        '''
        Args :
            X (np.ndarray) : X_test

        Returns :
            (np.ndarray) : probabilities for X_test
        
        '''
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict (self, X, threshold = 0.5) :
        '''
        Args :
            X (np.ndarray) : X_test

        Returns :
            (np.ndarray) : prediction for X_test 
        '''
        probability = self.predict_proba(X)

        y = []

        for prob in probability :
            if (prob >= threshold) : y.append(1)
            else : y.append(0)

        return np.array(y)

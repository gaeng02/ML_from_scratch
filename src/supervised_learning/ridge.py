import numpy as np
from logistic_regression import LogisticRegression

class Ridge (LogisticRegression) :
    
    def __init__ (self, learning_rate = 0.001, epochs = 500, l2_penalty = 0.1) :
        super().__init__(learning_rate, epochs)
        self.l2_penalty = l2_penalty
            
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

            dw = (1 / n_samples) * (np.dot(X.T, error) + self.l2_penalty * self.weights)
            self.weights -= self.learning_rate * dw

            db = (1 / n_samples) * np.sum(error)
            self.bias -= self.learning_rate * db

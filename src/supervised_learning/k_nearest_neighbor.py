import numpy as np
from src.distance import Euclidean

class KNN :

    def __init__ (self, k = 3) :
        self.k = k
    
    def fit (self, X, y) :
        self.X_train = X
        self.y_train = y

    def predict (self, X) :
        predictions = []

        for x in X :
            distances = []

            for x_train in self.X_train :
                distances.append(Euclidean(x, x_train))

            k_indixes = np.argsort(distances)[ : self.k]
            
            k_nearest_labels = []
            for index in k_indices :
                k_nearest_labels.append(self.y_train[i])

            labels, counts = np.unique(k_nearest_labels, return_counts = True)
            majority_label = labels[np.argmax(counts)]

            predictions.append(majority_label)

        return np.array(predictions)

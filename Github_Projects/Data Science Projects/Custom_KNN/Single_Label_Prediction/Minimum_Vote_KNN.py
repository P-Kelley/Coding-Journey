import numpy as np
import pandas as pd

#KNN classifier from https://combine.se/blog/ml-isnt-all-so-mysterious-implement-your-own-knn-classifier/ with some modifications for allowing a no label prediction and bug fix for kneighbor_labels in predict function . 
class KNNClassifer:
    #minvote is a added parameter that allows you to specify the minimum number of nearest neighbors with that label to vote for that label, otherwise it will vote for label -1, which will correspond to no label
    #The default value of 0 makes this act like a standard KNN algorithm
    def __init__(self, metric, n_neighbors = 5, minvote = 0):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.minvote = minvote

    def fit(self, X, y):
        self._X = X.copy().reset_index(drop = True)
        self._y = y.copy().astype(int)
        self.n_classes = len(set(y))



    def kneighbors(self, X, n_neighbors = None):
        """ 
        X: shape (n_queries, n_features)
        return: matrix of indices, shape(n_neighbors, n_queries)
        """
        n_neighbors = n_neighbors if n_neighbors else self.n_neighbors
        pairwise_distances = self.metric(self._X, X)
        return pairwise_distances.argsort(axis = 0)[:n_neighbors,]



    def predict(self, X):
        """
        X: shape (n_queries, n_features)
        y: shape (n_queries, )
        """
        
        kneighbor_indices = self.kneighbors(X)
        kneighbor_labels = np.apply_along_axis(
                lambda x : self._y.iloc[x],
                axis = 0,
                arr = kneighbor_indices
            )
        n_bins = self.n_classes

        #This creates 2D array with first array being number of zeros for each class and second array being number of 1s for each class. We will use the largest of the second
        label_counts = np.apply_along_axis(
            lambda column: np.bincount(column, minlength=n_bins),
            axis = 0,
            arr=kneighbor_labels
        )
        
    
        return np.where(np.max(label_counts, axis = 0) <= self.minvote, -1, label_counts.argmax(axis = 0))

    def score(self, X, y):
        """ 
        X: shape(n_samples, n_features)
        y: shape (n_samples, )
        returns score (float)
                Accuray of self.preict(X) wrt. y.
        """
        predictions = self.predict(X)
        n = len(y)
        return (predictions == y).sum() / n


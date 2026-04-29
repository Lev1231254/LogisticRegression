import numpy as np
import matplotlib.pyplot as plt


CLIP_VALUE = 500

class LogisticRegressionModel:
    def __init__(self, lr, batch_size, epochs, ridge_rate):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.mean = 0
        self.std = 1

        self.weights = np.array([])
        self.weights_history = []
        self.bias = 0

        self.ridge_rate = ridge_rate
    
    def normalize(self, features):
        return (features - self.mean) / self.std
    

    def fit(self, features_train, labels_train):

        # set normalization parameters
        self.mean = features_train.mean(axis=0)
        self.std = features_train.std(axis=0)

        # normalize features
        features_train = self.normalize(features_train)

        # set weights
        self.weights = np.random.randn(features_train.shape[1]) * 0.01

        # train model using gradient descent
        n = len(features_train)
        for _ in range(self.epochs):

            indices = np.random.permutation(n)
            features_train_copy = features_train[indices]
            labels_train_copy = labels_train[indices]

            for i in range(0, n, self.batch_size):
                features_batch = features_train_copy[i : i + self.batch_size]
                labels_batch = labels_train_copy[i : i + self.batch_size]

                self.update_weights(features_batch, labels_batch)
            
            self.weights_history.append(self.weights.copy())


    def update_weights(self, features_batch, labels_batch):
        n = len(features_batch)

        probs = self.calculate_probs(features_batch)

        dw = np.dot(features_batch.T, probs - labels_batch) / n + 2 * self.ridge_rate * self.weights
        db = np.sum(probs - labels_batch) / n
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db


    def calculate_probs(self, features_batch : np.array):
        features_num = features_batch.shape[1]
        log_odds = np.dot(features_batch, self.weights[:features_num]) + self.bias
        log_odds = np.clip(log_odds, -1 * CLIP_VALUE, CLIP_VALUE)  # prevents overflow
        return 1 / (1 + np.exp(-log_odds))
        
    

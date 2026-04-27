import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self, lr, batch_size, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.mean = []
        self.std = []

        self.weights = []
        self.bias = 0
    
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
            features_train = features_train[indices]
            labels_train = labels_train[indices]

            for i in range(0, n, self.batch_size):
                features_batch = features_train[i : i + self.batch_size]
                labels_batch = labels_train[i : i + self.batch_size]

                self.update_weights(features_batch, labels_batch)


    def update_weights(self, features_batch, labels_batch):
        n = len(features_batch)

        probs = self.calculate_probs(features_batch)

        dw = np.dot(features_batch.T, probs - labels_batch) / n
        db = np.sum(probs - labels_batch) / n
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db


    def calculate_probs(self, features_batch : np.array):
        features_num = features_batch.shape[1]
        log_odds = np.dot(features_batch, self.weights[:features_num]) + self.bias
        log_odds = np.clip(log_odds, -500, 500)  # prevents overflow
        return 1 / (1 + np.exp(-log_odds))


    def paint(self, features_test_raw, labels_test, ax, fig, heatmapIndices, features_names):
        features_test_norm = self.normalize(features_test_raw)

        # set features that we use
        i, j = heatmapIndices[0], heatmapIndices[1]

        x1raw = features_test_raw[:, i]
        x2raw = features_test_raw[:, j]

        x1norm = features_test_norm[:, i]
        x2norm = features_test_norm[:, j]

        extentRaw = (x1raw.min(), x1raw.max(), x2raw.min(), x2raw.max())
        extentNorm = (x1norm.min(), x1norm.max(), x2norm.min(), x2norm.max())        

        # ------plot decision boundary------
        
        ax[0].scatter(x1norm, x2norm, 
                      c=labels_test,
                      cmap='coolwarm')
        ax[0].set_xlabel('Normalized ' + str(features_names[i]))
        ax[0].set_ylabel('Normalized ' + str(features_names[j]))

        ax[0].set_title('Decision boundary')

        x_vals = np.linspace(x1norm.min(), x1norm.max(), 100)
        y_vals = -(self.weights[i] * x_vals + self.bias) / self.weights[j]

        ax[0].plot(x_vals, y_vals, color='black')
        
        # ------plot heatmap probabilities------
        probs = self.getHeatmapProbs(len(features_test_norm[0]), heatmapIndices, extentNorm)

        ax[1].imshow(probs, 
                     cmap='viridis',
                     extent=extentRaw,
                     origin='lower')
        
        ax[1].set_title('Probabilities')
        ax[1].set_xlabel(features_names[i])
        ax[1].set_ylabel(features_names[j])
        
    
    def getHeatmapProbs(self, num_of_features, features_id, extent):
        # make matrix of values
        x1min, x1max  = extent[0], extent[1]
        x2min, x2max = extent[2], extent[3]

        grid = []
        i, j = 0, 0

        for x in [x/10 for x in range(int(x1min * 10), int(x1max * 10), 1)]:
            grid.append([])
            j = 0
            for y in [y/10 for y in range(int(x2min * 10), int(x2max * 10), 1)]:
                grid[i].append([x, y])
                j += 1
            i += 1
            
        n, m = i, j
        grid =np.flip(np.array(grid), axis=0)
        
        #make probabilities matrix
        prob_grid = np.zeros((grid.shape[0], grid.shape[1]))
        i, j = 0, 0
        for i in range(n):
            for j in range(m):
                # calculate probability using only 2 features
                x1, x2 = grid[i][j][0], grid[i][j][1]

                features = np.array([[0. for _ in range(num_of_features)]])
                features[0][features_id[0]] = x1
                features[0][features_id[1]] = x2

                prob_grid[i][j] = self.calculate_probs(features)[0]
    
        return prob_grid
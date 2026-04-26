import numpy as np
import matplotlib.pyplot as plt

class logisticRegressionModel:
    def __init__(self, lr, batch_size, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.mean = []
        self.std = []

        self.weights = []
        self.bias = 0
    
    def normalize(self, ftrs):
        return (ftrs - self.mean) / self.std
    

    def fit(self, ftrs_train, labels_train):
        # set normalization parameters
        self.mean = ftrs_train.mean(axis=0)
        self.std = ftrs_train.std(axis=0)

        # normalize ftrs
        ftrs_train = self.normalize(ftrs_train)

        # set weights
        self.weights = np.random.randn(ftrs_train.shape[1]) * 0.01

        # train model using gradient descent
        n = len(ftrs_train)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            ftrs_train = ftrs_train[indices]
            labels_train = labels_train[indices]

            for i in range(0, n, self.batch_size):
                ftrs_batch = ftrs_train[i : i + self.batch_size]
                labels_batch = labels_train[i : i + self.batch_size]

                self.update_weights(ftrs_batch, labels_batch)


    def update_weights(self, ftrs_batch, labels_batch):
        n = len(ftrs_batch)

        probs = self.calculate_probs(ftrs_batch)

        dw = np.dot(ftrs_batch.T, probs - labels_batch) / n
        db = np.sum(probs - labels_batch) / n
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db


    def calculate_probs(self, ftrs_batch : np.array):
        ftrs_num = ftrs_batch.shape[1]
        log_odds = np.dot(ftrs_batch, self.weights[:ftrs_num]) + self.bias
        log_odds = np.clip(log_odds, -500, 500)  # prevents overflow
        return 1 / (1 + np.exp(-log_odds))


    def paint(self, ftrs_test_raw, labels_test, ax, fig, heatmapIndices, ftrs_names):
        # set labels for heatmap befor normalizing ftrs
        i, j = heatmapIndices[0], heatmapIndices[1]
        x1min, x1max = min(ftrs_test_raw[:, i]), max(ftrs_test_raw[:, i])
        x2min, x2max = min(ftrs_test_raw[:, j]), max(ftrs_test_raw[:, j])
        extent = (x1min, x1max, x2min, x2max)

        ftrs_test_norm = self.normalize(ftrs_test_raw)

        # plot decision boundary
        x1 = ftrs_test_raw[:, i]
        x2 = ftrs_test_raw[:, j]

        ax[0].scatter(x1, x2, 
                      c=labels_test,
                      cmap='coolwarm')
        ax[0].set_xlabel(ftrs_names[i])
        ax[0].set_ylabel(ftrs_names[j])

        ax[0].set_title('Decision boundary')

        x_vals = np.linspace(x1.min(), x1.max(), 100)
        y_vals = -(self.weights[i] * x_vals + self.bias) / self.weights[j]

        ax[0].plot(x_vals, y_vals, color='black')
        
        # plot heatmap probabilities
        
        # set min and max after normalization
        x1min, x1max = min(ftrs_test_norm[:, i]), max(ftrs_test_norm[:, i])
        x2min, x2max = min(ftrs_test_norm[:, j]), max(ftrs_test_norm[:, j])
        probs = self.getHeatmapProbs(len(ftrs_test_norm[0]), heatmapIndices, x1min, x1max, x2min, x2max)

        ax[1].imshow(probs, 
                     cmap='viridis',
                     extent=extent,
                     origin='lower')
        ax[1].set_title('Probabilities')
        
    def getHeatmapProbs(self, num_of_ftrs, ftrs_id, x1min, x1max, x2min, x2max):
        # make matrix of values

        grid = []
        i = 0
        j = 0
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
                # calculate probability using only 2 ftrs
                x1, x2 = grid[i][j][0], grid[i][j][1]

                ftrs = np.array([[0. for _ in range(num_of_ftrs)]])
                ftrs[0][ftrs_id[0]] = x1
                ftrs[0][ftrs_id[1]] = x2

                prob_grid[i][j] = self.calculate_probs(ftrs)[0]
    
        return prob_grid
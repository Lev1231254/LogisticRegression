import numpy as np
from logisticRegressionFromScratch import LogisticRegressionModel

def getHeatmapProbabilities(model, num_of_features, features_id, extent):
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

                prob_grid[i][j] = model.calculate_probs(features)[0]
    
        return prob_grid


def paint(model, features_test_raw, labels_test, ax, fig, heatmapIndices, features_names, titles_on):
        features_test_norm = model.normalize(features_test_raw)

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

        if titles_on: ax[0].set_title('Decision boundary')

        x_vals = np.linspace(x1norm.min(), x1norm.max(), 100)
        y_vals = -(model.weights[i] * x_vals + model.bias) / model.weights[j]

        ax[0].plot(x_vals, y_vals, color='black')
        
        # ------plot heatmap probabilities------
        probs = getHeatmapProbabilities(model, len(features_test_norm[0]), heatmapIndices, extentNorm)

        ax[1].imshow(probs, 
                     cmap='viridis',
                     extent=extentRaw,
                     origin='lower')
        
        if titles_on: ax[1].set_title('Probabilities')
        ax[1].set_xlabel(features_names[i])
        ax[1].set_ylabel(features_names[j])


        #-------plot weights history-------
        weights_history = np.array(model.weights_history)

        for i in range(len(features_names)):
            ax[2].plot(weights_history[:, i],
                       label=features_names[i])
            ax[2].legend()

        if titles_on: ax[2].set_title('Weights history')
        
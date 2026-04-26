from LogisticRegressionModel import logisticRegressionModel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')
iris_list = data.values.tolist()


# turn data into features and labels
features_train_setosa = np.array([iris[:-1] for iris in iris_list[0:40]])
features_test_setosa = np.array([iris[:-1] for iris in iris_list[40:51]])
features_train_versicolor = np.array([iris[:-1] for iris in iris_list[51:90]])
features_test_versicolor = np.array([iris[:-1] for iris in iris_list[90:101]])

features_train = np.concatenate([features_train_setosa, features_train_versicolor])
features_test = np.concatenate([features_test_setosa, features_test_versicolor])


labels_train_setosa = np.array([1 for iris in iris_list[0:40]]) 
labels_test_setosa = np.array([1 for iris in iris_list[40:51]])
labels_train_versicolor = np.array([0 for iris in iris_list[51:90]])
labels_test_versicolor = np.array([0 for iris in iris_list[90:101]])

labels_train = np.concatenate([labels_train_setosa, labels_train_versicolor])
labels_test = np.concatenate([labels_test_setosa, labels_test_versicolor])




model = logisticRegressionModel(0.1, 200, 1000)
model.fit(features_train, labels_train)

fig, ax = plt.subplots(2, 2)
model.paint(features_test, labels_test, ax[0], fig)
plt.show()
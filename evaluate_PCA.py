import numpy as np
import pandas as pd 

from sklearn import datasets, linear_model, preprocessing
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

digits = datasets.fetch_mldata("MNIST original")
X = digits.data
y = digits.target

X_train = X[:60000]
X_test = X[60000:70000]

y_train = y[:60000]
y_test = y[60000:70000]


PCA_model = PCA(n_components = 3, svd_solver = 'full').fit(X_train)
data_pca = PCA_model.transform(X_train)

reconstructed = PCA_model.inverse_transform(data_pca)

# mse = (np.square(result - X_train)).mean(axis=ax)

mse = mean_squared_error(reconstructed, X_train)/60000

fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()

colours = ['red','blue','green','orange','black','violet','cyan','gold','saddlebrown','lightpink']

for i in range(data_pca.shape[0]):
	x_val, y_val, z_val = data_pca[i]
	label = y_train[i]
	ax.scatter(x_val, y_val, z_val, c=colours[int(label)], s=20, marker = ('$'+str(int(label))+'$'))

plt.show()






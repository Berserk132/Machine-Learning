import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
#data = pd.read_csv(
#    'D:\PythonProjects\Machine-Learning\Assignment 2\Task 2\heart.csv')
data = pd.read_csv('heart.csv')


class SVM:

    def __init__(self, learningRate=0.001, lam = 0.01, iters=1000):
        self.w = None
        self.b = None
        self.lr = learningRate
        self.lam = lam
        self.iters = iters
        


    def update(self, X, y):
        numberOfSamples, numberOfFeatures = X.shape
        
        Y = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(numberOfFeatures)
        self.b = 0

        for _ in range(self.iters):
            for idx, x_i in enumerate(X):
                
                if  Y[idx] * (np.dot(x_i, self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lam * self.w)
                else:

                    self.w -= self.lr * (2 * self.lam * self.w - np.dot(x_i, Y[idx]))
                    self.b -= self.lr * Y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


data200 = data
yz = np.array(data200['target'])
for i in range(len(features)):
    for j in range(i+1, len(features)-1):
        df = pd.DataFrame(data200, columns=[features[i], features[j]])
        X = np.array(df)
        y = np.where(yz == 0, -1, 1)




        svm = SVM()
        svm.update(X, y)
        #predictions = svm.predict(X)

        print(svm.w, svm.b)

        def visualize_svm():
            def get_hyperplane_value(x, w, b, offset):
                return (-w[0] * x + b + offset) / w[1]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

            x0_1 = np.amin(X[:, 0])
            x0_2 = np.amax(X[:, 0])

            x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
            x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)

            x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
            x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

            x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
            x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)

            ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
            ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
            ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

            x1_min = np.amin(X[:, 1])
            x1_max = np.amax(X[:, 1])
            ax.set_ylim([x1_min-3, x1_max+3])

            plt.show()

        visualize_svm()

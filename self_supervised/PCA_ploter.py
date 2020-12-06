import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from sklearn import decomposition
from sklearn import datasets


class PCA_ploter():
    def __init__(self, X=0,y=0, save_name=None):
        # centers = [[1, 1], [-1, -1], [1, -1]]
        # iris = datasets.load_iris()
        # xx = iris.data
        # yy = iris.target
        np.random.seed(5)
        # fig = plt.figure(1, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        #
        # plt.cla()
        pca = decomposition.PCA(n_components=2)
        pca.fit(X)
        X = pca.transform(X)

        # for name, label in [('Normal', 0), ('raw image', 1)]:
            # ax.text2D(X[y == label, 0].mean(),
            #           X[y == label, 1].mean() + 1.5,
            #           X[y == label, 0].mean(), name,
            #           horizontalalignment='center',
            #           bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        # y = np.choose(y, [1, 2, 0]).astype(np.float)
        lx=len(X[:, 0])
        plt.cla()
        plt.scatter(X[:lx//2, 0], X[:lx//2, 1], c='b')
        plt.scatter(X[lx//2:, 0], X[lx//2:, 1], c='g')
        # plt.ion()
        # plt.show()
        plt.savefig(save_name)

# PCA_ploter()
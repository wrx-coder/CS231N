from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange
#版本兼容导入，无需在意

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
        #这个函数仅仅是保存下来数据

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]                    #X.shape[0] 表示 X 的第 0 维长度，也就是行数
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))  #准备一个全0矩阵准备填值
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                #soulution
                diff = X[i]-self.X_train[j]
                dists[i,j] = np.sqrt(np.sum(diff**2))
                #soulution
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            #soulution
            diff = X[i] - self.X_train
            dists[i,:] = np.sqrt(np.sum(diff**2,axis = 1)) #axis=1每行求和
            #solution
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        #直接作差大小不一样
        #对任意一对（测试样本x，训练样本y），距离平方 = x 平方 + y 平方 — 2 * 点积
        #点积：500*D D*5000 最后500*5000（可）
        #平方和：测试：500*1 训练5000*1，想要变成500*5000
        #故训练集转置为1*5000，然后相加，利用Numpy的广播和
        #按照行加和后默认变成一行数（类似1行N列），要保留数组形式，用Keepdims，变为N行1列
        #solution
        X_sq = np.sum(X ** 2, axis = 1, keepdims=True)
        train_sq = np.sum(self.X_train ** 2, axis = 1, keepdims=True).T
        cross = X.dot(self.X_train.T)
        dists = np.sqrt(X_sq + train_sq - 2*cross)
        #solution
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            #np.argsort:返回按距离排序后的索引（不是排序后的距离值）
            #solution
            nearest_idxs = np.argsort(dists[i])[:k]
            closest_y = self.y_train[nearest_idxs]
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            #solution
            counts = np.bincount(closest_y)#用 np.bincount 统计次数
            y_pred[i] = np.argmax(counts)  #np.argmax 找出现次数最多的标签
            #solution
        return y_pred

import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and labels to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0, 3.0] ] and the corresponding label would be [0, 1] label can be either a 0 or 1

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
#         print("self.features : ", self.features)
        self.labels = labels

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds the k nearest neighbours in the training set.
        It needs to return a list of labels of these k neighbours. When there is a tie in distance,
		prioritize examples with a smaller index.
        :param point: List[float]
        :return:  List[int]
        DO : p & f[0], p & f[1], p & f[2]
        DO will get the distances of p to each f
            Store distances in list [8, 2, 3, 11, 2] cooresponding indices [0, 1, 2, 3, 4]
            Sort indices of distances by smallest --> largest in list [1, 4, 2, 0, 3]
            k is how far to search in sorted indicis list
                k = 1, return [1]
                k = 3, return [1, 4, 2]
                k = 5, return [1, 4, 2, 0, 3]
        """
        store_dis = []
        self.store_dis = store_dis
#         compare our new point to points in features
        for feature in range(len(self.features)) :
#             print("point : ", point)
#             print("feature : ", self.features[feature])
#         distance is taking in our self.d_f func w/ comparing new points and features
            distance = self.distance_function(point, self.features[feature])
#             print("distance : ", distance)
            self.store_dis.append(distance)

        # indices from smallest to largest
        sort_dist = np.argsort(self.store_dis)[: self.k]
#         print("sort_dist : ", sort_dist)

#         return labels of the s_d
        return self.labels[sort_dist]

	# TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function.
            ==> features
        Here, you need to process every test data point
            ==> loop through every feature so len(self.features)
        reuse the get_k_neighbours function to find the nearest k neighbours for each test data point
            ==> get_k_neighbors(point)
        find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
            ==> label can be either a 0 or 1; given
            ==> features = [[25.0, 3.8], [22.0, 3.0], [21.0, 3.2], [21.0, 1.0], [15.0, 3.6], [22.0, 3.0]]
            ==> 0, 0, 0, 0, 1, 1 so majority labels would be 0
        Thus, you will get N predicted label for N test data point.
            ==> predicted label = test data point
            ==> add all 0s & 1s & that # should be the same as the total data points
        This function needs to return a list of predicted labels for all test data points.
            ==> return the labels of the features of the get_k_nei output (see below)
        :param features: List[List[float]]
        :return: List[int]
        """

        most_labels = []
#         for every feature
        for feature in range(len(features)) :
            get_k_n = self.get_k_neighbors(feature)
            # count 1s
            count_ones = (get_k_n == 1).sum()
#             print("count_ones : ", count_ones)
            # count 0s
            count_zeros = (get_k_n == 0).sum()
#             print("count_zeros : ", count_zeros)

    #         compare
            if  count_ones >  count_zeros :
                most_labels.append(1)
            else :
                most_labels.append(0)

        return most_labels


if __name__ == '__main__':
    print(np.__version__)

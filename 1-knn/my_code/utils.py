import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################

# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

# assert keyword tests if a condition is true
    assert len(real_labels) == len(predicted_labels)
#     raise NotImplementedError
#     1s : positive
#     0s: negative

    total_labels = len(real_labels) + len(predicted_labels)

#     get t_p : collect all the 1s in t_p
#     loop either r_l or p_l; same lenght so doesn't matter
    true_positive = 0
    for i in range(len(real_labels)) :
#         print(i, "real_labels : predicted_labels ==>", real_labels[i], predicted_labels[i])
#         ask : at this position, is there a 1;
        if (real_labels[i] == 1) and (predicted_labels[i] == 1) :
            true_positive += 1

#     get t_p_p :
    total_predicted_positive = 0
    for i in range(len(real_labels)) :
        if predicted_labels[i] == 1 :
            total_predicted_positive += 1

#     get f_n : evaluate both real & predicted
    false_negative = 0
    for i in range(len(real_labels)) :
#         print(i, "real_labels : predicted_labels ==>", real_labels[i], predicted_labels[i])
        if (real_labels[i] == 1) and (predicted_labels[i] == 0) :
            false_negative += 1
#     print("false_negative : ", false_negative)

    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0
    try:
        precision = true_positive / total_predicted_positive
    except ZeroDivisionError:
        precision = 0

    try:
        f1_score = ((precision * recall) / (precision + recall)) * 2
    except ZeroDivisionError:
        f1_score = 0

    return f1_score

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x_minus_x_prime = np.subtract(point1, point2)

        absolute_of_differece = np.absolute(x_minus_x_prime)

        cubed_diff = pow(absolute_of_differece, 3)

        sum_of_cubed_diff = np.sum(cubed_diff)

        cubed_root_of_sum = np.cbrt(sum_of_cubed_diff)

        return cubed_root_of_sum

    @staticmethod
    def euclidean_distance(point1, point2):
        x_minus_x_prime = np.subtract(point1, point2)

#         square difference
        squ_diff = pow(x_minus_x_prime, 2)

#         sum diff
        sum_of_squ_diff = np.sum(squ_diff)

#         square root the sum
        squ_root_of_sum = np.sqrt(sum_of_squ_diff)

        return squ_root_of_sum

    @staticmethod
    # TODO - refer to https://piazza.com/class/krcc1cltfj9264?cid=39
    def cosine_similarity_distance(point1, point2):

        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        addition_for_p1 = 0

        square_points = np.square(point1)
        addition_for_p1 = np.sum(square_points)
        # print("addition_for_p1 : ", addition_for_p1)

        square_points2 = np.square(point2)
        addition_for_p2 = np.sum(square_points2)
        # print("addition_for_p2 : ", addition_for_p2)


        numerator = np.sum(np.multiply(point1, point2))
        denom = np.multiply(np.sqrt(addition_for_p1), np.sqrt(addition_for_p2))


        if denom == 0 :
            return1_minus = 0
        else :
            return1_minus = 1 - (numerator / denom)
        if addition_for_p1 == 0 or addition_for_p2 == 0 :
            return  1
        else :
            return return1_minus



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        # raise NotImplementedError
        max_f1 = -1
#         if there are less than 30 points, can't have 30 neighbors
        bound = min(30, len(x_train))
        for name, dist_val in distance_funcs.items() :
            print(name, " : ", dist_val)

            for k_nums in range(1, bound, 2) :
#                 print("k : ", k)

                knn_m = KNN(k_nums, dist_val)
#                 print("knn : ", knn)

                knn_m.train(x_train, y_train)
#                 print("knn.train : ",  train)
#                 get f1 score & store them
#                 p = predict
                p = knn_m.predict(x_val)

#                 get the max f1
                f1 = f1_score(y_val, p)

                if f1 > max_f1 :
#                     new max
                    max_f1 = f1
                    self.best_k = k_nums
                    self.best_distance_function = name
                    self.best_model = knn_m

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        # raise NotImplementedError

        max_f1 = -1
#         if there are less than 30 points, can't have 30 neighbors
        bound = min(30, len(x_train))

        for name, dist_val in distance_funcs.items() :
            # print(name, " : ", dist_val)

            for s_c_name, s_c_val in scaling_classes.items() :
                # print(">>>>> ", s_c_name, " : ", s_c_val)
#             scale x_train & x_val
                s_c = s_c_val()
                # print("> ", name, " : x_train : ", x_train)

#                   pass in x_train as features
                scale_x_t = s_c(x_train)
                # print("scale_x_t : ", scale_x_t)

#                 print("x_val : ", x_val)
                scale_x_v = s_c(x_val)
#                 print("scale_x_v : ", scale_x_v)

                for k_nums in range(1, bound, 2) :
    #                 print("k : ", k)

                    knn_m = KNN(k_nums, dist_val)
    #                 print("knn_m : ", knn_m)

                    knn_m.train(scale_x_t, y_train)
    #                 print("knn.train : ",  train)
    #                 get f1 score & store them
    #                 p = predict
                    p = knn_m.predict(scale_x_v)

    #                 get the max f1
                    f1 = f1_score(y_val, p)

                    if f1 > max_f1 :
    #                     new max
                        max_f1 = f1
                        # print("max_f1 : ", max_f1)

                        self.best_k = k_nums
                        # print("self.best_k : ", self.best_k)

                        self.best_distance_function = name
                        # print("self.best_distance_function : ", self.best_distance_function)

                        self.best_scaler = s_c_name
                        print("self.best_scaler : ", self.best_scaler)

                        self.best_model = knn_m
                        # print("self.best_model : ", self.best_model)



class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        fs = np.asarray(features)
#         print(fs)

        normalized_vector = []
        normalized_point = []
        final_list = []

        for f in range(len(fs)) :
#             print("f : ", f)
            add_square_points = np.sum(np.square(fs[f]))
#             print("add_square_points : ", add_square_points)
            square_features = np.sqrt(add_square_points)
#             print("square_features : ", square_features)

#             print(fs[f], " / ", square_features)
            if all(fs[f] == 0) :
                normalized_point = []
                normalized_point.append(fs[f])
            else :
                normalized_point = []
                n = fs[f] / square_features
                normalized_point.append(n)

#             print("normalized_point : ", normalized_point)
            normalized_vector.append(normalized_point)
#         print("normalized_vector : ", normalized_vector)



        for list_n_v in range(len(normalized_vector)) :
            a = list(normalized_vector[list_n_v][0])

            float_n_v = [float(norm_vect) for norm_vect in a]
#             print("float_n_v : ", float_n_v)

            final_list.append(float_n_v)

        return final_list


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        # raise NotImplementedError
        fs = np.asarray(features)
        zeros = []
        non_zeros = []
        store_all = []
        final_list = []

        for f in range(fs.shape[1]):
            col = fs[:, f]
            get_min = np.min(col)
            get_max = np.max(col)

#             print("col ", f+1, col, get_min, get_max)
            if get_min == get_max :
                for c in range(len(col)) :
                    col[c] = float(0.0)
                    zeros.append(col[c])

                float_z_list = [float(a) for a in zeros]
#                 print("float_z_list : ", float_z_list)
                store_all.append(float_z_list)
            else :
#                 print("col : ", col)
                n = (col - get_min) / (get_max - get_min)
                non_zeros.append(n)

                for n_z in range(len(non_zeros)) :
#                     convert from array in list to list in list
                    non_zeros_list = list(non_zeros[n_z])
#                     print("non_zeros_list : ", non_zeros_list)

                    float_n_z_list = [float(a) for a in non_zeros_list]
#                     print("float_n_z_list : ", float_n_z_list)

                store_all.append(float_n_z_list)
#                 print("store_all : ", store_all)
        s_a_t = list(np.transpose(store_all))
#         print("s_a_t : ", s_a_t)

#         final_list = [list(item) for item in s_a_t[1]]


        for s in range(len(s_a_t)) :
#             print(s)
            l = list(s_a_t[s])
            final_list.append(l)

        return final_list

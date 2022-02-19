import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
#      => err = (1/N * summation of ((X^T*w) - y)^2) [see L2|P17,18]
    x_dot_w = np.dot(X, w)
    x_w_minus_y = np.subtract(x_dot_w, y)
    squ = np.square(x_w_minus_y)
    mean = np.mean(squ)
    err = mean

#     err = np.square(np.subtract(np.dot(X, np.transpose(w)), y)).mean()
    # print("err : ", err)
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here                    #
    #####################################################
#   w = (X^T*X)^-1 * X^Ty [see L2|P24,26]

    x_dot_x = np.dot(np.transpose(X), X)
    x_inv = np.linalg.inv(x_dot_x)
    x_dot_y = np.dot(np.transpose(X), y)

    w = np.dot(x_inv, x_dot_y)
    return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here                    #
    #####################################################
#    w* = inv(X^T*X + lambda*I)*X^T*y [see L2 | P32]
    cov = np.dot(np.transpose(X), X)
    lambd_dot_I = np.dot(lambd, np.identity(cov.shape[0]))
    cov_lambd = np.add(cov, lambd_dot_I)
    cov_lambd_inv = np.linalg.inv(cov_lambd)

    xt_y = np.dot(np.transpose(X), y)

    w = np.dot(cov_lambd_inv, xt_y)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################

    lambd_neg_i = -np.inf
    lambd_z = 0

    bestlambda_arr = []
    for x in range(0, 15) :
        bestlambda_arr.append(2**-x)

    global_min = -1
    mse = -1
    for x in range(len(bestlambda_arr)) :

        # print("\nXtrain shape : ", Xtrain.shape, "\nytrain shape : ", ytrain.shape)
        r_l_r = regularized_linear_regression(Xtrain, ytrain, bestlambda_arr[x])
        # print("\nXval shape : ", Xval.shape, "\nyval shape : ", yval.shape, "\n")
        mse = mean_square_error(r_l_r, Xval, yval)

        if mse < global_min :
            global_min = -1
            mse = -1
        else :
            bestlambda = bestlambda_arr[x]
            global_min = mse

    return bestlambda



###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################
    # check to see if it's squaring

    x_o = X
    N, D = X.shape

    # initialize the size of the new array
    new_X = np.array([[] for _ in range(N)])
    # print("new_X : ", new_X)
    for i in range(1, p+1) :

        # print("X : ", X.shape, X, "\np : ", p)
        x_p = np.power(x_o, i)
        # print("x_p : ", x_p)

        new_X = np.insert(new_X, [D * (i - 1)], x_p, axis=1)
    return new_X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2
    print("WE ARE PRINTING N", N)


    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      #
        ################################################
        # [L3 | P38]

        # if np.multiply(y, np.matmul(np.transpose(w), X)) + b <= 0: ind = 1
        # [float(i) for i in lst]
        # w = w + (step_size * ind * np.multiply(y, X))

        print("y : ", y.shape, "X : ", X.shape, "w : ", w.shape, "b : ", b)

        # set y values to -1s & 1s
        y = np.where(y == 0, -1, 1)
        print("y : ", y)

        # if point is correctly classified, then use continue or break
        # y :  (350,) X :  (350, 2) w :  (2,) b :  0
        for max_i in range(max_iterations + 1) :
            # print(w)
                X_w = np.dot(X, w)
                # print("x_w : ", x_w.shape)
                X_w_b = np.add(X_w, b)
                # print("X_w_b : ", X_w_b.shape)
                y_X_w_b = y * X_w_b
                # print("y_X_w_b : ", y_X_w_b.shape)
                indicator = np.where(y_X_w_b <= 0, 1, 0)
                # print("indicator : ", indicator.shape)
                i_y = np.multiply(indicator, y)
                # print("i_y : ", i_y.shape)
                i_y_X = np.dot(i_y, X)

                w = w + step_size * i_y_X / N
                # print("w : ", w)
                b = b + np.sum(step_size * i_y / N)
                # print("b : ", b)


    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    #
        ################################################

        # [L3 | P49]
        y = np.where(y == 0, -1, 1)
        for i in range (0, max_iterations) :
            X_w = np.dot(X, w)
            # print("X_w : ", X_w.shape)
            X_w_b = np.add(X_w, b)
            # print("X_w_b : ", X_w_b.shape)
            y_X_w_b = np.dot(y, X_w_b)
            # print("y_X_w_b : ", y_X_w_b.shape)

            # call sigmoid(z) function
            y_sigmoid = np.multiply(y,  sigmoid(-z))
            # print("y_sigmoid : ", y_sigmoid)

            grad_w = np.dot(np.transpose(X), y_sigmoid)
            # print("grad_w : ", grad_w.shape)
            grad_b = np.sum(y_sigmoid)
            # print("grad_b : ", grad_b.shape)

            w = w + step_size * grad_w / N
            # print("w : ", w.shape)
            b = b + step_size * grad_b / N
            # print("b : ", b.shape)

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b

# START HERE
def sigmoid(z):

    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value = 1 / (1 + np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model

    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape

    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    # wTx only for one point;  Xw for all points

    get_pred = np.dot(np.dot(X, np.transpose(w)), b)
    # print("get_pred : ", get_pred)
    if get_pred > 0 :
        return 1
    else :
        return -1

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass
    classification. Again for GD use the *average* of the gradients for all training
    examples multiplied by the step_size to update parameters.

    You may find it useful to use a special (one-hot) representation of the labels,
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":
        ones = np.ones((N, 1))
        get_X = np.append(X, ones, axis = 1)

        b = b.reshape((C,1))
        w = np.append(w, b, axis = 1)

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################
            X_n = get_X[n]
#            print()
            X_n = np.reshape(X_n, (D + 1, 1))
#            print(X_n.shape)
            # store in tuple
            lbs = np.zeros((1, C))
            lbs[0][y[n]] = 1
#            print('lbs', lbs.shape)
            w_X_n = np.dot(w, X_n)
            # print("w_X_n : ", w_X_n.shape)
            get_amax = np.amax(w_X_n)
            # print("max : ", max)

            minus = np.subtract(w_X_n - get_amax)
            # print("minus : ", minus)

            numer = np.exp(minus)
            # print("numer : ", numer)

            denom = np.sum(numer, axis = 0)
            # print("denom : ", denom)

            get_softmax = np.divide(numer, denom) - np.transpose(lbs)
            get_update = np.dot(get_softmax, np.transpose(x_n))

            w = w - step_size * get_update

        b = w[:, -1]
        w = w[:, :-1]


    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        pass


    else:
        raise "Undefined algorithm."


    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D
    - b: bias terms of the trained model, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    # wrap N x ones in tuple
    ones = np.ones((N, 1))

    classes = w.shape[0]

    get_X = np.append(X, ones, axis = 1)

    b = b.reshape((classes,1))
    # print("b : ", b.shape)

    w = np.append(w, b, axis = 1)
    # print("w : ", w.shape)

    get_w_X = np.dot(w, np.transpose(get_X))
    # print("z : ", z.shape)
    max_z = np.amax(max_z)

    numer = np.exp(get_w_X - max_z)
    # print("numer : ", numer)
    denom = np.sum(numer, axis = 0)
    # print("denom : ", denom)
    get_softmax = np.divide(numer, denom)
    # print("get_softmax : ", get_softmax)
    preds = np.argmax(get_softmax, axis = 0)

    assert preds.shape == (N,)
    return preds

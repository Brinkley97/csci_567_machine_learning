import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################

    center_idx = [p]
    center = [x[p]]
    distances = np.empty([len(x)])

    for k in range(n_cluster - 1) :

        for point in range(len(x)) :

            store_dists = []

            for c in center :

                dist = np.linalg.norm(x[point] - c)
                store_dists.append(dist)

            distances[point] = np.min(store_dists)

        prob = distances / np.sum(distances)
        sum_all = prob.cumsum()
        r = generator.rand()

        for j, p in enumerate(sum_all):
            if r < p:
                break
        center_idx.append(j)
        center.append(x[j])
    return center_idx

# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0),
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

#         index through x to get the centers
        self.centers = x[centroid_func(len(x), self.n_cluster, x, self.generator)]
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        prev_obj_func = None
        for c in range(self.max_iter) :


            points_in_clusters = [[] for _ in range(len(self.centers))]
            assignment = np.zeros([N])
            res = []
            obj_func = 0

            for d_p in range(len(x)) :
#                 axis = 1 to treat every cluster...
                dist = np.square(np.linalg.norm(x[d_p] - self.centers, axis=1))

                s_d_idx = np.argmin(dist)
                points_in_clusters[s_d_idx].append(x[d_p])
                assignment[d_p] = s_d_idx
#                 sum
                obj_func += dist[s_d_idx]
#             mean
            obj_func = obj_func/N
#           show points in each cluster
            for pt in range(len(points_in_clusters)) :
                if len(points_in_clusters[pt]) > 0 :
                    cluster_mean = np.mean(points_in_clusters[pt], axis=0)
#                     print("mean in cluster ", pt, cluster_mean)
                    self.centers[pt] = np.array(cluster_mean, dtype=float)

#             check convergence
            if prev_obj_func != None :
                if np.abs(obj_func - prev_obj_func) < self.e :
                    break
            prev_obj_func = obj_func
        return self.centers, assignment, c



class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        k_means = KMeans(self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = k_means.fit(x)

        centroid_labels = []

        # loop clusters
        for i in range(self.n_cluster):
            mem = y[(membership == i)]
            if (mem.size > 0):
                # Find the unique elements of an array
                _, idx, cts = np.unique(
                    mem, return_index=True, return_counts=True)

                get_max_idx = idx[np.argmax(cts)]
                index = get_max_idx
                centroid_labels.append(mem[index])
            else:
                centroid_labels.append(0)
        centroid_labels = np.array(centroid_labels)

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # print("N, D : ", N, D)
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        N = x.shape[0]
        # print("N : ", N)
        n_cluster = self.centroids.shape[0]
        dist = np.zeros((N, n_cluster))

        # loop clusters
        for c in range(n_cluster):
            dist[:, c] = np.square(np.sum(x - self.centroids[c], axis=1))

        cen_idx = np.argmin(dist, axis=1)
        # print(cen_idx)
        labs = []

        # get centroid index
        for c_i in cen_idx:
            labs.append(self.centroid_labels[c_i])
        return np.array(labs)




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################

    N, M = image.shape[:2]
    c_vec_shape = code_vectors.shape[0]
    # get distances
    dist = np.zeros((N, M, c_vec_shape))

    # loop over shape of code_vectors
    for i in range(c_vec_shape):
        dist[:, :, i] = np.square(np.sum((image - code_vectors[i]),  axis=2))

    idx = np.argmin(dist, axis=2)

    # set new img to min dist
    new_image = code_vectors[idx]
    return new_image

import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE

        mu_k = x[np.random.choice(N, self.n_cluster, replace=True),:]
        iteration, number_of_updates = 1, 0
        J = np.inf
        
        while iteration <= self.max_iter:

            distance_matrix = np.sum((x-np.expand_dims(mu_k, axis=1))**2, axis=2).T
            cluster_assignment = np.argmin(distance_matrix, axis=1)
            J_new = np.sum(distance_matrix.min(axis=1))/N
            number_of_updates += 1
            if np.abs(J - J_new) < self.e:
                break
            J = J_new
            
            mu_k = np.zeros((self.n_cluster,D))
            for l in range(self.n_cluster):
                tmp1 = np.sum(x[cluster_assignment == l,],axis = 0)
                tmp2 = np.sum(cluster_assignment == l)
                mu_k[l,:] = tmp1/tmp2
                # print("tmp1: ", tmp1.shape, "tmp2: ", tmp2, "x.shape: ", x.shape, "cluster: ", self.n_cluster)
            
            iteration += 1 
        
        return mu_k, cluster_assignment, number_of_updates
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = k_means.fit(x)
        centroid_labels = []
        
        for cluster in range(self.n_cluster):
            # its member
            sub_y = y[membership == cluster]
            (_, idx, counts) = np.unique(sub_y, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            centroid_labels.append(sub_y[index])
        centroid_labels = np.array(centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        pred = []
        for m in range(N):
            distances = []
            for n in range(self.n_cluster):
                sub = np.subtract(x[m,:],self.centroids[n,:])
                distances.append(np.inner(sub,sub))
            pred.append(self.centroid_labels[np.argmin(distances)])
            
        labels = np.array(pred)

        # distance_matrix = np.sum((x-np.expand_dims(self.centroids, axis=1))**2, axis=2).T
        # cluster_assignment = np.argmin(distance_matrix, axis=1)

        # DONOT CHANGE CODE BELOW THIS LINE
        return labels


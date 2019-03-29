import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            ini_k_means = KMeans(n_cluster=self.n_cluster, max_iter = self.max_iter, e = self.e)
            mu_k, membership, num_updates = ini_k_means.fit(x)
            
            variance_k = []
            pi_k = []
            for v in range(self.n_cluster):
                x_temp = x[membership == v,:]
                subtract = x_temp - mu_k[v]
                var_cov = np.dot(subtract.T, subtract)
                variance_k.append(var_cov/x_temp.shape[0])
                pi_k.append(x_temp.shape[0]/N)
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            mu_k = np.random.uniform(low=0.0, high=1.0, size=(self.n_cluster,D))
            variance_k  = [np.identity((D))]*self.n_cluster
            pi_k = [1/self.n_cluster]*self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        self.means, self.variances,self.pi_k = np.array(mu_k),np.array(variance_k),np.array(pi_k)
            
        l = self.compute_log_likelihood(x)
        
        iteration = 1
        while iteration <= self.max_iter:
            ##--- E step: compute responsibilities
            gamma_ik_temp = np.zeros((N,self.n_cluster))
            pdfs = []
            for u in range(self.n_cluster):
                pdfs.append(self.Gaussian_pdf(mu_k[u],variance_k[u]))

            for p in range(N):
                for u in range(self.n_cluster):
                    gamma_ik_temp[p,u] = pi_k[u]*pdfs[u].getLikelihood(x[p])

            gamma_ik = np.divide(gamma_ik_temp,np.transpose(np.tile(np.sum(gamma_ik_temp,axis = 1),(self.n_cluster,1))))
            
            N_k = np.sum(gamma_ik,axis = 0)

            ##--- M step
            #------ Estimate means
            mu_k = []
            for s in range(self.n_cluster):
                mu_k.append(np.sum(np.multiply(np.transpose(np.tile(gamma_ik[:,s],(D,1))),x),axis = 0)/N_k[s])

            #----Estimate Variance, mixturn-weight
            variance_k = []
            pi_k = []
            for m in range(self.n_cluster):
                var = np.zeros((D,D))
                for t in range(N):
                    minus = np.reshape((x[t] - mu_k[m]),(D,1))
                    var = var + gamma_ik[t,m]*np.dot(minus,np.transpose(minus))
                variance_k.append(var/N_k[m])
                pi_k.append(N_k[m]/N)
            
            self.means, self.variances,self.pi_k = np.array(mu_k),np.array(variance_k),np.array(pi_k)
            #---- converged? loglikelihood
            l_new = self.compute_log_likelihood(x)
            
            if np.abs(l - l_new) <= self.e:
                break
            iteration = iteration + 1
            l = l_new
        return iteration
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        mu_k,variance_k,pi_k = self.means, self.variances,self.pi_k
        draw_sample = []
        for a in range(N):
            k = np.argmax(np.random.multinomial(1, pi_k, size=1))
            draw_sample.append(np.random.multivariate_normal(mu_k[k], variance_k[k]))
            
        samples= np.array(draw_sample)
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE

        n_cluster = self.n_cluster
        px = []
        N,D= x.shape

        pdfs = []
        for u in range(n_cluster):
            pdfs.append(self.Gaussian_pdf(means[u], variances[u]))

        for n in range(N):
            pxi = 0
            for u in range(n_cluster):
                tmp = pdfs[u].getLikelihood(x[n,:])
                pxi = pxi + pi_k[u]*tmp
            px.append(pxi)

        loglike = np.sum(np.log(px)).astype(float)
        log_likelihood = np.float64(loglike).item()
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None:
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to (sqrt（(2pi)^D) * det(variance)） (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE

            while np.linalg.matrix_rank(self.variance) < self.variance.shape[0]:
                self.variance += 1e-3*np.eye(self.variance.shape[0])

            self.inv = np.linalg.inv(self.variance)
            D = self.mean.shape[0]
            self.c = np.sqrt(np.power(2*np.pi, D)*np.linalg.det(self.variance))
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            
            K = x.shape[0]
            tmp = np.reshape((x - self.mean),(K,1))
            part1 = np.exp(-1/2*np.dot(np.dot(tmp.T,self.inv),tmp))
            p = part1/self.c

            # DONOT MODIFY CODE BELOW THIS LINE
            return p

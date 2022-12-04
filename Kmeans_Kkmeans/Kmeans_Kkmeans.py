"""
 *
 *Created by Ismail BACHCHAR
 """

#imports
import numpy as np
import copy

def kmeans(X, k, max_iter, history=False):
    
    """
    Arguments:
        X : data matrix
        k : number of clusters
        max_iter : maximum number of iterations to converge
        history : if True return the centroids updates norm-differences
        
    """
    
    #number of samples
    n =X.shape[0]
    
    #init
    centroids = X[np.random.choice(n, k, replace = False)] ##randomly choose k data points as initial centroids
    prev_centroids = np.zeros((k, X.shape[1]))
    clusters = [0 for _ in range(n)] #indices representing cluster of data points
    history = [0 for _ in range(max_iter)]

    for iter in range(max_iter):
        
        history[iter] = np.linalg.norm(centroids - prev_centroids)
        if history[iter] != 0:
            
            for i in range(n):
                #assign clusters
                clusters[i] = np.argmin(np.sqrt(np.sum((X[i] - centroids)**2, axis=1)))
                
            prev_centroids = copy.deepcopy(centroids)
            for i in range(k):
                #update centroids
                centroids[i] = np.mean([X[j] for j in range(n) if clusters[j] == i], axis=0)
                
                # don't update the centroid if the cluster has no points
                if np.isnan(centroids[i]).any():
                    centroids[i] = prev_centroids[i]
        
        else:
            history = history[:iter]
            break
        
    if history is True: return centroids, clusters, history
    else: return centroids, clusters

def predict(X, centroids):
    centroids_ = []
    clusters = []
    for x in X:
        dists = np.sqrt(np.sum((x - centroids)**2, axis=1)) #point dists to diff centroids
        centroid_index = np.argmin(dists) #nearest cluster
        clusters.append(centroid_index)
        centroids_.append(centroids[centroid_index]) #centroids of nearest cluster
    
    return centroids_, clusters
    
    
def kernalize_f(X, kernel, par1=None, par2=None):
    
    """
    kernalizing X
    Arguments:
        X : is the data matrix/vector
        kernel : the kernel to be used, must be in ['linear', 'rbf', 'sigmoid', 
            'polynomial', 'cosine', 'laplacian']
        par1 and par2 represent the different parameters used by each kernel

    Return:
        the gram matrix
    """
    # number of sample
    n = X.shape[0]
    n_matrix = np.ones((n, n))/n

    if kernel=='linear':
        c = 2
        if par1 is not None:
            c = par1
        kernel_dist = (X@X.T) + c

    elif kernel=='rbf':
        gamma = 0.06
        if par1 is not None:
            gamma = par1
   
        kernel_dist = np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis = 2)**2)
       
    elif kernel=='sigmoid':
        gamma = 0.06
        if par1 is not None:
            gamma = par1
        
        c = 2
        if par2 is not None:
            c = par1

        kernel_dist = np.tanh(gamma * X@X.T + c)

    elif kernel=='polynomial':
        degree = 3
        if par1 is not None:
            degree = par1

        kernel_dist = (X@X.T)**degree

    elif kernel=='cosine': 
        kernel_dist = (X@X.T/np.linalg.norm(X, 1) * np.linalg.norm(X, 1))

    elif kernel=='laplacian':
        gamma = 0.06
        if par1 is not None:
            gamma = par1

        kernel_dist = np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis = 2))
    else:
        print("The chosen kernel is not supported")
        return
    # return kernel_dist ###########
    kernelized_X = kernel_dist - n_matrix@kernel_dist - kernel_dist@n_matrix + n_matrix@kernel_dist@n_matrix
    return kernelized_X

def calculate_dist(KX, n, k, clusters, dist):
    
    """
    Arguments:
        KX : kernalized data matrix
        k : number of clusters
        dist : data point to cluster distance matrix  
    """
    
    dp_weights = np.ones(n) #data points weights
    
    for i in range(k):
        
        mask = clusters==i
        if np.sum(mask)==0:
            # Empty cluster was found! we don't update previous clusters/dists
            return dist
        
        KX_i = KX[mask][:, mask]
        denom = np.sum(dp_weights[mask])
        denom_squared = denom*denom
        dist_i = np.sum(KX_i) / denom_squared
        # within_dists[i] = dist_i
        
        dist[:, i] += (dist_i - 2*np.sum(KX[:, mask], axis=1)) / denom
        
        # print("cluster of")

    return dist
    
def kernel_kmeans(X, k, max_iter, kernel, halt=1e-3, par1=None, par2=None):
    
    """
    Arguments:
        X : data matrix
        k : number of clusters
        max_iter : maximum number of iterations to converge
        kernel : the kernel to be used, must be in ['linear', 'rbf', 'sigmoid', 
            'polynomial', 'cosine', 'laplacian']
        halt : tolerance
        
    """
    
    #number of samples
    n =X.shape[0]
        
    KX = kernalize_f(X, kernel) #nxn matrix
    
    #init
    dist = np.zeros((n, k)) #data point to cluster distance
    clusters = np.random.randint(k, size=n) #indices representing cluster of data points
    # within_dists = np.zeros(k)

    
    for iter in range(max_iter):
        #update data to cluster distances
        dist = calculate_dist(KX, n, k, clusters, dist)
        
        prev_clusters = copy.deepcopy(clusters)
        
        #assign points to clusters
        clusters = dist.argmin(axis=1)
        #samples that their clusters stayed the same
        n_unchanged = float(np.sum((clusters - prev_clusters) == 0))
        
        if 1 - n_unchanged / n < halt:
            break
        
    return clusters
         
def predict(X, k, ):
    KX = kernalize_f(X) #nxn matrix
    n = X.shape[0]
    dist = np.zeros((n, k))
    dist = calculate_dist(KX, k, dist)
    
    return dist.argmin(axis=1)
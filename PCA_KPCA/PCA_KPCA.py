"""
 *
 *Created by Ismail BACHCHAR
 """

#imports
import numpy as np
from numpy.linalg import svd 


def pca(X, r, project=True):

    """
    Arguments:
        X : is the data matrix
        r : is the number of principal components (PC) to keep
        project : if true retunr projected data otherwise return the PCs

    Return:
        projected data or principal components depending on variable 'project'
        explained_variance : explained variance by every component

    """

    #number of samples
    n =X.shape[0]
    #number of dimensions
    d = X.shape[1]

    if r>d:
        print("r must be smaller or equal to number of features/dimensions")
        return

    #center X
    X -= np.mean(X, axis=0)

    #svd of centered data
    U, D, V_t = svd(X, full_matrices=False) #singular values are sorted be default
    #eigen_vals = (D**2)/(n-1)

    #first r principal components
    PCs = V_t.T[:, :r]
    # PCs[:, -1] = PCs[:, -1]*-1

    #explained variance
    eigen_values = (D**2)/(n-1)
    sum_eigens = np.sum(eigen_values)
    explained_variance = [x/sum_eigens*100 for x in sorted(eigen_values, reverse = True)[:r]]

    if project is True:
        #projected data into the first r PCs
        X_projected = X@PCs
        # X_appx = np.dot(np.dot(U, np.diag(D))[:, :r], V_t[:r]) #using only first r rows of V_t
        return X_projected, explained_variance
    else:
        return PCs, explained_variance

def gram_matrix_f(X1, X2, kernel, par1=None, par2=None):
    
    """
    Calculate the Gramm Matrix between X1 and X2 using the chosen kernel function
    Arguments:
        X1 : is the data matrix/vector
        X2 : is the data matrix/vector
        kernel : the kernel to be used, must be in ['linear', 'rbf', 'sigmoid', 
            'polynomial', 'cosine', 'laplacian']
        par1 and par2 represent the different parameters used by each kernel

    Return:
        the gram matrix
    """
    # number of sample
    n = X1.shape[0]
    n_matrix = np.ones((n, n))/n

    if kernel=='linear':
        c = 0
        if par1 is not None:
            c = par1
        kernel_dist = (X1@X2.T) + c

    elif kernel=='rbf':
        gamma = 1
        if par1 is not None:
            gamma = par1
   
        kernel_dist = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis = 2)**2)
       
    elif kernel=='sigmoid':
        gamma = 0.008
        if par1 is not None:
            gamma = par1
        
        c = 0
        if par2 is not None:
            c = par1

        kernel_dist = np.tanh(gamma * X1@X2.T + c)

    elif kernel=='polynomial':
        degree = 4
        if par1 is not None:
            degree = par1

        kernel_dist = (X1@X2.T)**degree

    elif kernel=='cosine': 
        kernel_dist = (X1@X2.T/np.linalg.norm(X1, 1) * np.linalg.norm(X2, 1))

    elif kernel=='laplacian':
        gamma = 0.008
        if par1 is not None:
            gamma = par1

        kernel_dist = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis = 2))
    else:
        print("The chosen kernel is not supported")
        return
    # return kernel_dist ###########
    gram_matrix = kernel_dist - n_matrix@kernel_dist - kernel_dist@n_matrix + n_matrix@kernel_dist@n_matrix
    return gram_matrix

def kpca(X, r, kernel='linear', transform=True, par1=None, par2=None):

    """
    Arguments:
        X : is the data matrix
        kernel : the kernel to be used, must be in ['linear', 'rbf', 'sigmoid', 'polynomial', 'cosine']
        r : is the number of principal components (PC) to keep
        transform : if true retunr transformed data otherwise return the PCs
        par1 and par2 represent the different parameters used by each kernel

    Return:
        transformed data or principal components depending on variable 'transform'
        explained_variance : explained variance by every component

    """
    
    #number of samples
    n =X.shape[0]
    #number of dimensions
    d = X.shape[1]

    if r>d:
        print("r must be smaller or equal to number of features/dimensions")
        return

    gram_matrix = gram_matrix_f(X1=X, X2=X, kernel=kernel, par1=par1, par2=par2)
    # return gram_matrix ###########

    eigen_vals, eigen_vects = np.linalg.eig(gram_matrix)
    eigen_vals_sorter = np.argsort(eigen_vals[:r])[::-1]
    
    sum_eigens = np.sum(eigen_vals)
    explained_variance = [x/(sum_eigens) for x in sorted(eigen_vals, reverse = True)[:r]]

    #sort eigen_vals and corresponding eigen_vects in case they are not returned sorted
    eigen_vals, eigen_vects = eigen_vals[:r], eigen_vects[:, eigen_vals_sorter]
    #first r components
    eigen_vals, eigen_vects = eigen_vals[:r], eigen_vects[:, :r]
    PCs = eigen_vects.T

    if transform is True:
        #transformed data into the first r PCs
        X_transformed = gram_matrix@eigen_vects
        return X_transformed, explained_variance
    else:
        return PCs, explained_variance

"""
 *
 *Created by Ismail BACHCHAR
 """

#imports
import scipy as sp

import sys
sys.path.append('../PCA_KPCA/')
import PCA_KPCA

from sklearn import decomposition


def mmd(source, target):
    
    """
    Arguments:
        source : first domain data (distribution P)
        target : to compare with domain data (distribution Q)

    Return:
        experiment results as dictionary using all datasets

    """
    
    #means calculation
    source_mean = source@source.T
    target_mean = target@target.T
    st_mean = source@target.T
    
    #mmd distance calculation
    return source_mean.mean(axis=None) + target_mean.mean(axis=None) - 2*st_mean.mean(axis=None)

def mmd_original_space():

    data_names = ['CaffeNet4096', 'GoogleNet1024', 'surf']
    domains = ['Caltech10', 'amazon', 'webcam', 'dslr']

    results = {'CaffeNet4096': dict(), 'GoogleNet1024': dict(), 'surf': dict()}

    for data_name in data_names:
        distances = {'Caltech10': [], 'amazon': [], 'webcam': [], 'dslr': []}

        for i in range(4):
            #source domain
            temp = sp.io.loadmat(f'../data/office_caltech/{data_name}/{domains[i]}.mat')
            source_fts = temp['fts']
            # source_labels = temp['labels']
            for j in range(4):
                if i==j:
                    continue
                
                #target domain
                temp = sp.io.loadmat(f'../data/office_caltech/{data_name}/{domains[j]}.mat')
                target_fts = temp['fts']
                # target_labels = temp['labels']
                
                #means calculation
                source_mean = source_fts@source_fts.T
                target_mean = target_fts@target_fts.T
                st_mean = source_fts@target_fts.T
                
                #mmd distance calculation
                mmd = source_mean.mean(axis=None) + target_mean.mean(axis=None) - 2*st_mean.mean(axis=None)
                distances[domains[i]].append(mmd)
        results[data_name] = distances
    return results


def mmd_pca(r=[2], sklearn=False):
    
    """
    Arguments:
        r : list of PCA dimension
        sklearn : if True, use scikitLearn implementation of PCA otherwise use ours

    Return:
        experiment results as dictionary using all datasets

    """
    
    data_names = ['CaffeNet4096', 'GoogleNet1024', 'surf']
    domains = ['Caltech10', 'amazon', 'webcam', 'dslr']

    results = {'CaffeNet4096': dict(), 'GoogleNet1024': dict(), 'surf': dict()}

    for data_name in data_names:
        distances = {'Caltech10': dict(), 'amazon': dict(), 'webcam': dict(), 'dslr': dict()}
        for i in range(4):
            #source domain
            source_fts = sp.io.loadmat(f'../data/office_caltech/{data_name}/{domains[i]}.mat')['fts']
            if source_fts.dtype!='float64':
                source_fts = source_fts.astype('float64')
            
            for j in range(4):
                if i==j:
                    continue
                
                #target domain
                target_fts = sp.io.loadmat(f'../data/office_caltech/{data_name}/{domains[j]}.mat')['fts']
                if target_fts.dtype!='float64':
                    target_fts = target_fts.astype('float64')
                
                #project data using pca
                for ri in r:
                    if sklearn:
                        source = decomposition.PCA(n_components=ri).fit_transform(source_fts)
                        target = decomposition.PCA(n_components=ri).fit_transform(target_fts)
                    else:
                        source, _ = PCA_KPCA.pca(source_fts, r=ri, project=True)
                        target, _ = PCA_KPCA.pca(target_fts, r=ri, project=True)
                    
                    source_mean = source@source.T
                    target_mean = target@target.T
                    st_mean = source@target.T
                
                    #mmd distance calculation
                    mmd = source_mean.mean(axis=None) + target_mean.mean(axis=None) - 2*st_mean.mean(axis=None)
                    try:
                        distances[domains[i]][ri].append(mmd)
                    except KeyError:
                        distances[domains[i]][ri] = [mmd]
                
            results[data_name] = distances
    return results


def mmd_kpca(r=[2], kernel='rbf', par1=None, par2=None):
    
    """
    Arguments:
        r : list of PCA dimension
        kernel : the kernel to be used, must be in ['linear', 'rbf', 'sigmoid', 'polynomial', 'cosine']
        par1 and par2 represent the different parameters used by each kernel

    Return:
        experiment results as dictionary using all datasets

    """
    
    data_names = ['CaffeNet4096', 'GoogleNet1024', 'surf']
    domains = ['Caltech10', 'amazon', 'webcam', 'dslr']

    results = {'CaffeNet4096': dict(), 'GoogleNet1024': dict(), 'surf': dict()}

    for data_name in data_names:
        distances = {'Caltech10': dict(), 'amazon': dict(), 'webcam': dict(), 'dslr': dict()}
        for i in range(4):
            #source domain
            source_fts = sp.io.loadmat(f'../data/office_caltech/{data_name}/{domains[i]}.mat')['fts']
            if source_fts.dtype!='float64':
                source_fts = source_fts.astype('float64')
            
            for j in range(4):
                if i==j:
                    continue
                
                #target domain
                target_fts = sp.io.loadmat(f'../data/office_caltech/{data_name}/{domains[j]}.mat')['fts']
                if target_fts.dtype!='float64':
                    target_fts = target_fts.astype('float64')
                
                #project data using pca
                for ri in r:
                    #cannot run our implementation of kpca due to memory limits
                    # source, _ = PCA_KPCA.kpca(X=source_fts, r=ri, kernel=kernel, transform=True, par1=par1, par2=par2)
                    # target, _ = PCA_KPCA.kpca(X=target_fts, r=ri, kernel=kernel, transform=True, par1=par1, par2=par2)
                    source = decomposition.KernelPCA(n_components=ri, kernel=kernel, gamma=par1, degree=par2).fit_transform(source_fts)
                    target = decomposition.KernelPCA(n_components=ri, kernel=kernel, gamma=par1, degree=par2).fit_transform(target_fts)
                    
                    source_mean = source@source.T
                    target_mean = target@target.T
                    st_mean = source@target.T
                
                    #mmd distance calculation
                    mmd = source_mean.mean(axis=None) + target_mean.mean(axis=None) - 2*st_mean.mean(axis=None)
                    try:
                        distances[domains[i]][ri].append(mmd)
                    except KeyError:
                        distances[domains[i]][ri] = [mmd]
                
            results[data_name] = distances
    return results
"""
 *
 *Created by Ismail BACHCHAR
 """

#imports
from metric_learn import LMNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../MMD/')
import mmd

def mmd_dists(X_iris, y_iris, X_wine, y_wine):
    
    results = {'iris': dict(), 'wine': dict()}
    for dti, dt in enumerate([[X_iris, y_iris], [X_wine, y_wine]]):
        distances = {0: [], 1: [], 2: []}
        for i in range(3):
            #source domain
            source = dt[0][dt[1]==i]
            t = []
            for j in range(3):
                if i==j:
                    continue
                target = dt[0][dt[1]==j]
                t.append(mmd.mmd(source, target))
            distances[i] = t
            
        results[list(results.keys())[dti]] = distances
        
    return results

def mmd_dists_after_lmnn(X_iris, y_iris, X_wine, y_wine, k=1):
    
    results = {'iris': dict(), 'wine': dict()}
    
    lmnn = LMNN(k=k, learn_rate=1e-6)
    XX_iris = lmnn.fit_transform(X_iris, y_iris)
    XX_wine = lmnn.fit_transform(X_wine, y_wine)
    
    for dti, dt in enumerate([[XX_iris, y_iris], [XX_wine, y_wine]]):
        distances = {0: [], 1: [], 2: []}
        for i in range(3):
            #source domain
            source = dt[0][dt[1]==i]
            t = []
            for j in range(3):
                if i==j:
                    continue
                target = dt[0][dt[1]==j]
                t.append(mmd.mmd(source, target))
            distances[i] = t
            
        results[list(results.keys())[dti]] = distances
        
    return results

def score_1NN(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def score_1NN_after_lmnn(X_train, y_train, X_test, y_test, k=1):
    
    lmnn = LMNN(k=k, learn_rate=1e-6)

    XX_train = lmnn.fit_transform(X_train, y_train)

    return score_1NN(XX_train, y_train, X_test, y_test)

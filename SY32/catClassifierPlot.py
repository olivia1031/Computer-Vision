# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:12:43 2020

@author: xuphilip
"""
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

class CatClassifier():
    """ Classifieur de chats 
    
    On distingue deux types de chat : type A (-1) et type B (+1).
    On classifie les chats en fonction de leur poids x à l'aide du classifier
    f_h défini telle que : f_h(x) = -1 si x <= h et +1 sinon.
    
    Parameters
    ----------
    
    Attributes
    ----------
    h_hat : float
            Valeur optimale de h.
    
    """
    X_train = np.loadtxt('SY32_P20_TD01_data_X.csv', ndmin=2)
    y_train = np.loadtxt('SY32_P20_TD01_data_y.csv')
    h_hat = -np.Inf
    
    def __init__(self):
        pass
    
    def predict(self, X, h=None):
        """
        Prédit le type de chat pour les poids X et le paramètre h
        
        Parameters
        ----------
        X : array-like of shape (n_sample, n_features)
            Poids des chats
        
        h : float (default=h_hat)
            Paramètre de décision 
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            Type des chats
        """
        if h is None:
            h = self.h_hat
        # TODO
        ###append
        '''
        y = np.array([])
        for x in np.nditer(X, order = 'C'): 
            if x <= h:
                y = np.append(y, -1)
            else:
                y = np.append(y, 1)  
        return y  
        '''
        ### pre allocate
        '''
        y = np.zeros(len(X))
        for i, x in enumerate(X):
        if x <= h:
            y[i] = -1
        else:
            y[i] = 1
        return y       
        '''
        ### return 
        '''
        return [-1 if x<=h h else 1 for x in X]
        '''

        # sans boucle
        # t = time.time()
        return ((X>h)*2-1).flatten()
        # time = time.time() - t
    
    def err_emp(self, X, y, h=None):
        """
        Calcule l'erreur empirique de f_h sur l'ensemble (X,y).

        Parameters
        ----------
        X : array-like of shape (n_sample, n_features)
            Poids des chats
            
        y : array-like of shape (n_samples,)
            Type des chats
            
        h : float (default=h_hat)
            Paramètre de décision 

        Returns
        -------
        erreur : float
                 Erreur empirique

        """
        if h is None:
            h = self.h_hat
        
        # TODO
        y_pred = self.predict(X, h)
        TP = np.sum(y_pred!=y)
        N = len(y)
        return TP/N
    
    def fit(self, X, y):
        """
        Calcule la valeur optimale de h sur l'ensemble (X,y).
        L'attribut h_hat est mis à jour.'
        
        Parameters
        ----------
        X : array-like of shape (n_sample, n_features)
            Poids des chats
        
        y : array-like of shape (n_samples,)
            Type des chats
        
        Returns
        -------
        self : object
        """
        
        # TODO
        # the result change only when we pass a point
        # space = np.arange(np.min(silf.X_train), np.max(self.X_train), 0.01)
        # for h in space:
        err_min = 1
        for h in np.nditer(X):
            err = self.err_emp(X, y, h)
            if err < err_min:
                err_min = err
                self.h_hat = h
        return self
# nlog(n) (faire un trie, resolutino opitmal)

# K-Fold Cross Validation
# sublist
# masque: M=[true,true..]
# evaluer la performance de la methode
# k-fold: k by order
# shuffle split: random
# stratifiedKFord: same proportion of each class
# stratified shuffle split: random min each class
    def CV(self, X, y, K):
        l = len(y)
        unit = l//K
        Err = 0
        for k in np.arange(K):
            s = k*unit
            if(k < K-1):
                x_valide = np.take(X, [s, s+unit])
                y_valide = np.take(y, [s, s+unit])
                x_train = np.delete(X, [s, s+unit])
                y_train = np.delete(y, [s, s+unit])            
            else:
                x_valide = X[s:]
                y_valide = y[s:]
                x_train = X[0:s]
                y_train = y[0:s]
            self.fit(x_train, y_train)
            te = self.err_emp(x_valide, y_valide)
            Err += te
        return Err/K


    def cv_sk(self, X, y, K):
        kf = KFold(n_splits=K)
        kf.get_n_splits(self.X_train)
        Err = 0
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.fit(X_train, y_train)
            te = self.err_emp(X_test, y_test)
            Err += te
        return Err/K

    def afficher_cross_vaidation(self, X, y):
        err = []
        for k in range(2, 11):
            err.append(self.cv_sk(X, y, k))
        plt.plot(range(1, 10), err)
        plt.show()

    def err_theo(self, h, mu1=4, sigma1=1, mu2=6, sigma2=1, p1 =1/3):
        p2 = 1-p1
        return (1-norm.cdf(h, mu1, sigma1))*p1 + norm.cdf(h, mu2, sigma2)*p2

#%%
import numpy as np
import matplotlib.pyplot as plt

X_train = np.loadtxt('SY32_P20_TD02_data_X_train.csv', ndmin=2)
y_train = np.loadtxt('SY32_P20_TD02_data_y_train.csv')
h_hat = -np.Inf

class CatClassifierMultiDim():
    
    d_hat = 0;
    z_hat = -1;
    
    def predict(self, X, h=None, d=None, z=None):
        if h is None:
            h = self.h_hat
        if d is None:
            d = self.d_hat
        if z is None:
            z = self.z_hat
    
        y= np.full(X.shape[0],-z)
        y[X[:,d] <= h] = z
        
        return y
        pass
    
    def err_emp(self, X,y, h=None, d=None, z=None):
        if h is None:
            h = self.h_hat
        if d is None:
            d = self.d_hat
        if z is None:
            z = self.z_hat
        
        # TODO
        y_pred = self.predict(X,h,d,z)
        TP = np.sum(y_pred!=y)
        N = len(y)
        return TP/N
    
    
    def fit(self, X, y):
        err_min = np.Inf
        for z in [-1,1]:
            for d in range(X.shape[1]):
                for h in X[:,d]:
                    err = self.err_emp(X,y,h,d,z)
                    if err<err_min:
                        err_min = err
                        self.h_hat = h
                        self.d_hat = d
                        self.z_hat = z
        return self
        pass

 
    
def plotCatClassifier(clf, xlim=(0,10), ylim=(40, 56), xstep=0.1, ystep=0.1):
    # Génère une grille d'instances de X
    x = np.arange(xlim[0], xlim[1], xstep)
    y = np.arange(ylim[0], ylim[1], ystep)
    x_ = np.repeat(x, len(y))
    y_ = np.tile(y, len(x))
    X = np.column_stack((x_,y_))
    # Prédit la classe de X
    Y = clf.predict(X)
    # Trace le graphe de prédiction
    plt.figure()
    plt.plot(X[Y<0,0], X[Y<0,1], 'rs')
    plt.plot(X[Y>0,0], X[Y>0,1], 'bs')
    plt.legend(('Type A', 'Type B'))
    plt.show()
    
clf = CatClassifierMultiDim()
clf.fit(X_train, y_train)

plotCatClassifier(clf)
    
#%%

class CatClassifierMultiDim2():
    
    d_hat = 0;
    z_hat = -1;
    
    def predict(self, X, h=None, d=None, z=None):
        if h is None:
            h = self.h_hat
        if d is None:
            d = self.d_hat
        if z is None:
            z = self.z_hat
    
        y= np.full(X.shape[0],-z)
        y[X[:,d] <= h] = z
        
        return y
        pass
    
    def err_emp(self, X,y, h=None, d=None, z=None, p=None):
        if h is None:
            h = self.h_hat
        if d is None:
            d = self.d_hat
        if z is None:
            z = self.z_hat
        if p is None:
            p = 1
        
        # TODO
        y_predict = self.predict(X,h,d,z)
        mask = y_predict != y
        err = np.sum(mask*p)
        return err
    
    
    def fit(self, X, y, p=None):
        err_min = np.Inf
        for z in [-1,1]:
            for d in range(X.shape[1]):
                for h in X[:,d]:
                    err = self.err_emp(X,y,h,d,z)
                    if err<err_min:
                        err_min = err
                        self.h_hat = h
                        self.d_hat = d
                        self.z_hat = z
        return self
        pass


class CatClassifierBoost():
    
    liste_clfs = []
    liste_alphas = []
    
    def predict(self, X, clfs=None, alphas=None):
        if clfs is None：:
            clfs = self.liste_clfs
        if alphas is None :
            alphas = self.liste_alphas
        assert len(clfs) == len(alphas)
        K= len(clfs)
        assert K>0
        
        a_akfk = np.zeros((X.shape[0],K))
        fro k in range(K):
            fk = clfs[k].predict(X)
            a_akfk[:,k] = fk*alphs[k]
            
        y = np.full((X.shape[0]),-1)
        y[np.sum(a_akfk,aixs=1)>0] =1
        return y
    
    def fit(self,X,y,K):
        self.liste_clfs = []
        self.liste_alphas = []
        
        n = X.shape[0]
        K = 0
        p = np.full((n),1./n)
        
        while k<K:
            clf = CatClassifierMultiDim2()
            clf.fit(X,y,p)
            err = clf.err_emp(X, y, p=p)
            
            alpha = (1./2.)*np.log((1-err)/err)
            
            p = (p*np.exp(-alpha*y*clf.predict(X)))/ (2.*np.sqrt(err*(1-err)))
            
            k += 1
            self.liste_clfs.append(clf)
            self.liste_alphas.append(alpha)
            

   
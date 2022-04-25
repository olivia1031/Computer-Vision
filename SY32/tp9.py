# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:22:02 2021

@author: ooo
"""

from skimage.feature import hog
from skimage import io, util


i = 1
I = io.imread('imageface/train/pos/%05d.png'%i) #00001
I = util.img_as_float(I)
#de convertir les valeurs des pixels à valeurs entières {0, 1, . . . , 255} en valeurs flottantes [0, 1]
io.imshow(I)

#%% Ex2

import numpy as np

X_train  =np.zeros((15000,576))
for i in range(3000):
    I = io.imread('imageface/train/pos/%05d.png'%(i+1))
    I = util.img_as_float(I)
    X_train[i,:] = I.flatten()
for i in range(12000):
    I = io.imread('imageface/train/neg/%05d.png'%(i+1))
    I = util.img_as_float(I)
    X_train[i+3000,:] = I.flatten()
    
y_train = np.full((15000),-1)
y_train[:3000] = 1

#%% Ex4
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=10)
clf.fit(X_train,y_train)

#%% Ex5

X_test  =np.zeros((6256,576))
for i in range(1000):
    I = io.imread('imageface/test/pos/%05d.png'%(i+1))
    I = util.img_as_float(I)
    X_test[i,:] = I.flatten()
for i in range(5256):
    I = io.imread('imageface/test/neg/%05d.png'%(i+1))
    I = util.img_as_float(I)
    X_test[i+1000,:] = I.flatten()
    
y_test = np.full((6256),-1)
y_test[:1000] = 1

#%%
y = clf.predict(X_test)

err = np.mean(y!=y_test)
print("err= "+str(err))

#%%

clf2 = AdaBoostClassifier(n_estimators=50)
clf2.fit(X_train,y_train)
y2 = clf2.predict(X_test)

err2 = np.mean(y2!=y_test)
print("err2= "+str(err2))

#%% KNN

from sklearn import neighbors

n_neighbors = 5

clf_knn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf_knn.fit(X_train,y_train)

y_knn = clf_knn.predict(X_test)

err_knn = np.mean(y_knn!=y_test)
print("err_knn= "+str(err_knn))

#%% arbre de decision
from sklearn import tree

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_train,y_train)

y_tree = clf_tree.predict(X_test)

err_tree = np.mean(y_tree!=y_test)
print("err_tree= "+str(err_tree))

#%% foret aleatoire
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=10)
clf_RF.fit(X_train,y_train)

y_RF = clf_RF.predict(X_test)

err_RF = np.mean(y_RF!=y_test)
print("err_RF= "+str(err_RF))

#%% SVM
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X_train,y_train)

y_svm = clf_svm.predict(X_test)

err_svm = np.mean(y_svm!=y_test)
print("err_svm= "+str(err_svm))

#%% Histogram of Oriented Gradients
from skimage.feature import hog
I = io.imread('imageface/train/pos/00001.png')
X = hog(I)

#%%
X_hog_train = np.zeros((15000,81))

for i in range(X_train.shape[0]):
    Ihog = hog(X_train[i,:].reshape((24,24)))
    X_hog_train[i,:] = Ihog

X_hog_test = np.zeros((6256,81))

for i in range(X_test.shape[0]):
    Ihog = hog(X_test[i,:].reshape((24,24)))
    X_hog_test[i,:] = Ihog
    
#%% AdaBoost sur HOG

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=10)
clf.fit(X_hog_train,y_train)

y = clf.predict(X_hog_test)

err_adaHOG10 = np.mean(y!=y_test)
print("err_adaHOG10= "+str(err_adaHOG10))

clf2 = AdaBoostClassifier(n_estimators=50)
clf2.fit(X_hog_train,y_train)

y2 = clf2.predict(X_hog_test)

err_adaHOG50 = np.mean(y2!=y_test)
print("err_adaHOG50= "+str(err_adaHOG50))

#%% HOG KNN

from sklearn import neighbors

n_neighbors = 5

clf_knn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf_knn.fit(X_hog_train,y_train)

y_knn = clf_knn.predict(X_hog_test)

err_knnHOG = np.mean(y_knn!=y_test)
print("err_knnHOG= "+str(err_knnHOG))

#%% HOG arbre

from sklearn import tree

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_hog_train,y_train)

y_tree = clf_tree.predict(X_hog_test)

err_treeHOG = np.mean(y_tree!=y_test)
print("err_treeHOG= "+str(err_treeHOG))


#%% HOG foret aleatoire
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=10)
clf_RF.fit(X_hog_train,y_train)

y_RF = clf_RF.predict(X_hog_test)

err_RFHOG = np.mean(y_RF!=y_test)
print("err_RFHOG= "+str(err_RFHOG))

#%% HOG SVM
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X_hog_train,y_train)

y_svm = clf_svm.predict(X_hog_test)

err_svmHOG = np.mean(y_svm!=y_test)
print("err_svmHOG= "+str(err_svmHOG))
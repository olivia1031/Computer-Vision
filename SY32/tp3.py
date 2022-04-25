# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:16:47 2021

@author: ooo
"""

#%% Ex1.3
import numpy as np
from skimage import io
from skimage import color
from skimage import util
from scipy.ndimage import filters
import matplotlib.pyplot as plt

%matplotlib auto


I1 = io.imread('E:\\360MoveData\\Users\\ooo\\Desktop\\SY32\\s3\\taxi15.png')
I2 = io.imread('E:\\360MoveData\\Users\\ooo\\Desktop\\SY32\\s3\\taxi16.png')

I1 = I1.astype('float')
I2 = I2.astype('float')

D = I2-I1

Gx = 0.5 * np.array([[1,0,-1]])
Gy = Gx.T

gradx = filters.convolve(I1,Gx)
grady = filters.convolve(I1,Gy)

fig,axes= plt.subplots(1,4)
axes[0].imshow(I1,cmap='gray')
axes[0].set_title("I1" )
axes[1].imshow(D,cmap='gray')
axes[1].set_title("D" )
axes[2].imshow(gradx,cmap='gray')
axes[2].set_title("gradient x" )
axes[3].imshow(grady,cmap='gray')
axes[3].set_title("gradient y" )
plt.show()

#%% Ex1.5

def depl1point(x,y,N,gradx,gray,delta):
    # estimer le vecteur de deplacement de point(x,y)
    # N taille de la fenetre autoue de ce point(doit etre impair)
    # delta : image des differences entre les trames successives
    
    # en cas ou x et y sont floats
    x = int(round(x))
    y = int(round(y))
    
    #gerer le fenetre
    L = N//2
    view_gradx = gradx[y-L:y+L+1,x-L:x+L+1]
    view_grady = grady[y-L:y+L+1,x-L:x+L+1]
    view_delta = delta[y-L:y+L+1,x-L:x+L+1]
    #construire A
    A = np.hstack((view_gradx.reshape((-1,1)),view_grady.reshape((-1,1))))
    #construire B
    B = -view_delta.reshape((-1,1))# 1 colone, auto ligne
    #construire d
    d = np.linalg.inv(A.T @ A ) @ A.T @ B
    return d
    
N = 17

plt.figure()
plt.imshow(I1,cmap='gray')
pts = plt.ginput(4,0)

plt.show()

for pt in pts:
    depl = depl1point(pt[0],pt[1],N,gradx,grady,D)
    print(depl)
    # pour l'affichage, faire *10 du deplacement
    depl*=10
    plt.arrow(pt[0],pt[1],depl[0,0],depl[1,0],head_width=1,length_includes_head=True,color='r')


#%% Ex1.6

# flot optique image complete
# generer une matrice 3D avec (y,x,2 dims pour depl)
F = np.zeros((I1.shape[0],I1.shpe[1],2),dtype='float')

for x in range(N//2,I1.shpe[1]-N//2):
    for y in range(N//2,I1.shpe[0]-N//2):
        depl = depl1point(x,y,N,gradx,grady,D)
        F[y,x,:] = depl.flatten()
    
# affichage
plt.figure()
plt.imshow(I1,cmap='gray')

plt.show()

for x in range(N//2,I1.shpe[1]-N//2,8):
    for y in range(N//2,I1.shpe[0]-N//2,8):
        depl = F[y,x,:].copy()
        depl *= 10
        plt.arrow(x,y,depl[0],depl[1],head_width=1,length_includes_head=True,color='r')
 
#%% Ex1.7

# segmentation sur l'amplitude des deplacement

# calcul des amplitude
ampl = np.zeros(I1.shape,dtype='float')
ampl = np.sqrt(np.sum(F**2,aixs=2))

plt.figure()
plt.imshow(ampl)

plt.colorbar()
plt.show()

seuil = 0.7

M = np.zeros(I1.shape,dtype='uint8')
M[ampl>seuil]=255

plt.figure()
plt.imshow(M,cmap='gray')

plt.show()

I1seg = I1.copy().astype('uint8')
I1seg[M==0] = 0

plt.figure()
plt.imshow(I1seg,cmap='gray')

plt.show()
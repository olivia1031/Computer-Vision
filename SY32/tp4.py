# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:18:17 2021

@author: ooo
"""
import numpy as np
from skimage import io
from skimage import color
from skimage import util
from scipy.ndimage import filters
import matplotlib.pyplot as plt

%matplotlib auto

#%% Ex2.1
from scipy.interpolate import griddata

def transformer(I,H,hw=(-1,-1),interp='linear'):
    h = hw[0]
    w = hw[1]
    if (w<=0 or h<= 0):
        (h,w)=hw=I.shape[:2] #image.shape:height,weight,color
    O = np.zeros((h,w)) #image de sortie
    
    #np.arange(3)=>[1,2,3]
    #meshgrid: Build a coordinate matrix
    xx1,yy1 = np.meshgrid(np.arange(I.shape[1]),np.arange(I.shape[0]))
    xx1 = xx1.flatten()#flatten:une dimension
    yy1 = yy1.flatten()
    #print("xx1"+str(xx1)) #xx1[  0   1   2 ... 509 510 511]
    Hinv = np.linalg.inv(H)
    
    xx2,yy2 = np.meshgrid(np.arange(O.shape[1]),np.arange(O.shape[0]))
    xx2 = xx2.flatten()
    yy2 = yy2.flatten()
    #print("xx2"+str(xx2)) #xx2[  0   1   2 ... 509 510 511]

    xxyy2 = np.stack((xx2,yy2,np.ones((O.size))),axis=0)
    '''P2=[xi]
          [yi]
           [1]   coordonnees homogenes'''
    # P2=H*P1 
    #ici xxyy est l'image destinaire apres translation donc comme le location
    # puis chercher les pixels de Image 1 par interpolation pour remplir cet image
    xxyy = Hinv @ xxyy2
    xxyy = np.stack((xxyy[0]/xxyy[2],xxyy[1]/xxyy[2]),axis=0)
# =============================================================================
#     print("xxyy"+str(xxyy))
#     print("xxyy2"+str(xxyy2))
# =============================================================================
    # xxyy back to espace euclidien
    O = griddata((xx1,yy1),I.flatten(),xxyy.T,method=interp,fill_value=0).reshape(O.shape)
    # interpolation bilinear
    return O

from skimage.data import camera
im = camera( ).astype('float')

H = np.array([[1,0,10],
              [0,1,20],
              [0,0,1]])
im2 = transformer(im,H)

fig,axes= plt.subplots(1,2)
axes[0].imshow(im,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(im2,cmap='gray')
axes[1].set_title("I2" )
plt.show()

#%% Ex2.2
H1 = np.array([[1,0,100],
               [0,1,100],
               [0,0,1]])

H2 = np.array([[0.4,0,0],
               [0,0.4,0],
               [0,0,1]])

H3 = np.array([[1,0,-100],
               [0,1,-100],
               [0,0,1]])
H = H1 @ H2 @ H3
im3 = transformer(im,H)

fig,axes= plt.subplots(1,2)
axes[0].imshow(im,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(im3,cmap='gray')
axes[1].set_title("I3" )
plt.show()
#%% Ex2.3
import math
cos = math.cos(np.deg2rad(20))
sin = math.sin(np.deg2rad(20))
H4 = np.array([[cos,sin,0],
               [-sin,cos,0],
               [0,0,1]])
H = H1 @ H4 @ H3
im4 = transformer(im,H)

fig,axes= plt.subplots(1,2)
axes[0].imshow(im,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(im4,cmap='gray')
axes[1].set_title("I4" )
plt.show()

#%% Ex3
from skimage import transform
from skimage.data import camera
I = camera( ).astype('float')
h,w = np.shape(I)[:2]

coinsI = np.array([[0,0],[w,0],[w,h],[0,h]])
#dans l’ordre haut-gauche, haut-droit, bas-droit, puis bas-gauche //(0,0)est haut gauche
O = np.zeros((512,512),dtype='uint8')
#Créer une image de destination initialisée à zéro

plt.figure()
plt.imshow(I,cmap='gray')
plt.axis('off')
coinsO = np.array(plt.ginput(4,0))#choisir 4 points
plt.show()

tform = transform.estimate_transform('projective',coinsI,coinsO)
H = tform.params
#extraire les paramètres de transformation géométrique sous la forme d’une matrice d’homographie H 
print(H)

O = transformer(I,H,hw=O.shape[:2])

fig,axes= plt.subplots(1,2)
axes[0].imshow(I,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(O,cmap='gray')
axes[1].set_title("O" )
plt.show()

# plt.figure()
# plt.subplot(121)
# plt.imshow(I,cmap='gray')
# plt.axis('off')
# plt.subplot(121)
# plt.imshow(O,cmap='gray')
# plt.plot(coinsO[:,0],coinsO[:,1],'rx')
# plt.axis('off')
# plt.show()

#%% EX1
from skimage.color import rgb2gray
from scipy.interpolate import griddata
I1 = io.imread('E:\\360MoveData\\Users\\ooo\\Desktop\\SY32\\s4\\oldtown.png')
I1 = I1.astype('float')
#convertir en niveaux de gris pour faciliter les traitements
I = rgb2gray(I1)
# plt.figure()
# plt.imshow(I,cmap='gray')
# plt.show()

def cart2polaire(I):
    #centre(x0,y0)
    x0 = I.shape[1]/2
    y0 = I.shape[0]/2
       
    xx1,yy1 = np.meshgrid(np.arange(I.shape[1]),np.arange(I.shape[0]))
    #matrices xx1 et yy1 représentant les coordonnées dans cette image
    xx1 = xx1.flatten()
    yy1 = yy1.flatten()
    
    O = np.zeros((250,360),dtype='float') #image de sortie
    #deux matrices xx2 et yy2 représentant les coordonnées dans l’image de destination
    xx2,yy2 = np.meshgrid(np.arange(O.shape[1]),np.arange(O.shape[0]))
    xx2 = xx2.flatten()
    yy2 = yy2.flatten()
    
    # interprétant xx2 comme un angle (entre 0 et 360 degrés) //image destinaire width 360
    # yy2 comme un rayon par rapportau centre (x0, y0)
    # calculer les coordonnées correspondantes (xx,yy)
    xx = (yy2*np.cos(xx2*np.pi/180.))+x0
    yy = (yy2*np.sin(xx2*np.pi/180.))+y0
    
    O = griddata((xx1,yy1),I.flatten(),(xx,yy),method='linear',fill_value=0).reshape((-1,360))
    O = np.flip(O,0) #inverser l'axe Y
    return O

I2 = cart2polaire(I)

fig,axes= plt.subplots(1,2)
axes[0].imshow(I,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(I2,cmap='gray')
axes[1].set_title("I2" )
plt.show()


#%% EX1.5
I1 = io.imread('E:\\360MoveData\\Users\\ooo\\Desktop\\SY32\\s4\\oldtown.png')
IR,IG,IB=I1[:,:,0],I1[:,:,1],I1[:,:,2]

IR2 = cart2polaire(IR)
IG2 = cart2polaire(IG)
IB2 = cart2polaire(IB)
Ifusion=np.stack([IR2,IG2,IB2],axis=-1)

fig,axes= plt.subplots(1,5)
axes[0].imshow(IR2,cmap='gray')
axes[0].set_title("IR2" )
axes[1].imshow(IG2,cmap='gray')
axes[1].set_title("IG2" )
axes[2].imshow(IB2,cmap='gray')
axes[2].set_title("IB2" )
axes[3].imshow(Ifusion)
axes[3].set_title("Ifusion" )
axes[4].imshow(I1)
axes[4].set_title("I1" )
plt.axis('off')
plt.show()

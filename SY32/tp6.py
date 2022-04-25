# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:12:43 2021

@author: ooo
"""
import numpy as np
from skimage import io
from skimage import color
from skimage import util
from scipy.ndimage import filters
import matplotlib.pyplot as plt

%matplotlib auto

#%%

Ig = io.imread('E:\\360MoveData\\Users\\ooo\\Desktop\\SY32\\SY32_P21_TD06_fichiers\\stereo-pics\\tsukuba\\tsukuba1.png')
Id = io.imread('E:\\360MoveData\\Users\\ooo\\Desktop\\SY32\\SY32_P21_TD06_fichiers\\stereo-pics\\tsukuba\\tsukuba3.png')

# =============================================================================
# Ig = Ig.astype('float')
# Id = Id.astype('float')
# =============================================================================

fig,axes = plt.subplots(1,2,sharex=True,sharey=True)
axes[0].imshow(Ig)
axes[0].set_title("Ig")
axes[1].imshow(Id)
axes[1].set_title("Id")
plt.show()

#%% Ex3
# ce code effectue du block matching couleur en cumulant les dissimilarités de tous les canaux
def blockmatching(Iref, Irech, mindisp, maxdisp, taillefen) :
    # gestion padding avec du noir
    mitaillefen = taillefen // 2
    Irefp = np.zeros((Iref.shape[0]+2*mitaillefen, Iref.shape[1]+2*mitaillefen, 3), dtype='float32')
    Irechp = np.zeros((Irech.shape[0]+2*mitaillefen, Irech.shape[1]+2*mitaillefen, 3), dtype='float32')
    Irefp[mitaillefen:mitaillefen+Iref.shape[0], mitaillefen:mitaillefen+Iref.shape[1]] = Iref
    Irechp[mitaillefen:mitaillefen+Irech.shape[0], mitaillefen:mitaillefen+Irech.shape[1]] = Irech
    #ajout simple de bords noirs pour pouvoir calculer des disparités sur toute l'image sans erreur de code
    
    # initialisation de la carte des disparités
    carte_disp = np.zeros(Iref.shape[:2], dtype='int16')
    
    for y in range(Iref.shape[0]):
        for xref in range(Iref.shape[1]):
            fenref = Irefp[y:taillefen+y, xref:taillefen+xref]
            min_dissimilarite = np.inf #infinity
            disparite = 0
            for xrech in range(max(xref+mindisp, 0),min(xref+maxdisp, Irech.shape[1])):
                fenrech = Irechp[y:taillefen+y, xrech:taillefen+xrech]
                dissimilarite = np.sum(np.fabs(fenref - fenrech)) # SAD
                #dissimilarite = np.sum((fenref - fenrech)**2) # SSD
                if dissimilarite < min_dissimilarite:
                    min_dissimilarite = dissimilarite
                    disparite = xrech-xref
            carte_disp[y,xref] = disparite
            
    return carte_disp
#%%
taillefen3 = 3
taillefen5 = 5
taillefen7 = 7
maxdisp = 30
carte_disp_g3 = blockmatching(Ig, Id, -maxdisp, 0, taillefen3)
carte_disp_g5 = blockmatching(Ig, Id, -maxdisp, 0, taillefen5)
carte_disp_g7 = blockmatching(Ig, Id, -maxdisp, 0, taillefen7)

fig,axes = plt.subplots(1,3,sharex=True,sharey=True)
axes[0].imshow(carte_disp_g3)
axes[0].set_title("carte_disp_g3")
axes[1].imshow(carte_disp_g5)
axes[1].set_title("carte_disp_d5")
axes[2].imshow(carte_disp_g7)
axes[2].set_title("carte_disp_d7")
plt.show()
    

#%% Ex4 
taillefen = 5
maxdisp = 30
carte_disp_g = blockmatching(Ig, Id, -maxdisp, 0, taillefen)
carte_disp_d = blockmatching(Id, Ig, 0, maxdisp, taillefen) 

fig,axes = plt.subplots(1,2,sharex=True,sharey=True)
axes[0].imshow(carte_disp_g)
axes[0].set_title("carte_disp_g")
axes[1].imshow(carte_disp_d)
axes[1].set_title("carte_disp_d")
plt.show()
    

#%% Ex5
def mode_filter(I, size):
    # initialiser la sortie
    O = np.copy(I)
    # pas de padding => ne pas traiter les bords, plus simple
    misize = size // 2
    for y in range(misize,I.shape[0]-misize):
        for x in range(misize,I.shape[1]-misize):
            region = I[y-misize:y-misize+size, x-misize:x-misize+size]
            tri = np.sort(region, axis=None)
# =============================================================================
#             print("tri:")
#             print(tri) #array 25 of disparite                           
# =============================================================================
            cardmax = 0
            disp_cardmax = 0
            n = 0
            while n < tri.size:
                cardinalite = np.count_nonzero(tri == tri[n])
                if cardinalite > cardmax:
                    disp_cardmax = tri[n]
                    cardmax = cardinalite
                n += cardinalite
# =============================================================================
#             values, counts = np.unique(tri, return_counts=True)
#             return values[np.argmax(counts)]
# =============================================================================
            O[y,x] = disp_cardmax 
            #choos the max count value of array #un filtre médian
            #print("disp:"+str(disp_cardmax))
    return O

img = mode_filter(carte_disp_g, 5)

fig,axes = plt.subplots(1,2,sharex=True,sharey=True)
axes[0].imshow(carte_disp_g)
axes[0].set_title("carte_disp_g")
axes[1].imshow(img)
axes[1].set_title("img")
plt.show()

#%% Ex7
# prédiction image droite à partir de gauche et disparités

Idp = np.full_like(Ig, 127)

xx, yy = np.meshgrid(np.arange(Ig.shape[1]), np.arange(Ig.shape[0]))

Idp[yy,xx+carte_disp_g] = Ig

Edp = np.sqrt(np.sum((Idp.astype('float') - Id.astype('float'))**2,axis=2))

fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
axes[0].imshow(Id)
axes[0].set_title("Id")
axes[0].axis('off')
axes[1].imshow(Idp)
axes[1].set_title("Idp")
axes[1].axis('off')
axes[2].imshow(Edp, cmap='gray')
axes[2].set_title("Erreur Idp")
axes[2].axis('off')
plt.show()
    
    
    
    
    
    
    

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:23:42 2021

@author: ooo
"""

import tensorflow
from tensorflow import keras 

from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')

#%%
print(model.summary())
#afficher architecture du reseau de neurones

#%%
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

cougar = load_img('cougar.jpg',target_size=(224,224))
crab = load_img('crab.jpg',target_size=(224,224))
kangaroo = load_img('kangaroo.jpg',target_size=(224,224))

img = img_to_array(cougar)
#print(img[0,0,:])
img = np.expand_dims(img,axis=0)
img = preprocess_input(img)

img2 = img_to_array(crab)
img2 = np.expand_dims(img2,axis=0)
img2 = preprocess_input(img2)

img3 = img_to_array(kangaroo)
img3 = np.expand_dims(img3,axis=0)
img3 = preprocess_input(img3)

#%%
import matplotlib.pyplot as plt
%matplotlib auto

plt.figure()
plt.imshow(kangaroo)

#%%
y = model.predict(img)
#prédit la classe de l'image sous la forme d'un vecteur de score sur un ensemble 1 000 classes.

#%%
from tensorflow.keras.applications.vgg16 import decode_predictions

label = decode_predictions(y)
#retrouver les noms des classes ayant les scores les plus élevés.
print(label)
0
#%% EX6
'''
la dernière couche du réseau de neurones sert à prédire la classe de l'image. 
Les couches intermédiaires font parti du processus d'extraction de caractéristiques. 
Pour récupérer la représentation de l'image sur l'avant dernière couche fc2, 
on peut construire le modèle suivant :
'''
from tensorflow.keras.models import Model
model_feat = Model(inputs=model.input , outputs=model.get_layer('fc2').output)

#%%
'''
La commande x = model_feat.predict(img) permet alors de récupérer un vecteur de dimension 4096 représentant l'image. 
Reprendre les images du TD précédent et utiliser le réseau VGG16 pour trouver, 
pour chacune des images cougar.jpg, crab.jpg et kangaroo.jpg, les images les plus proches parmi les 300 images de la base.
'''
nbimgs = 300
allfeat = np.empty((nbimgs,4096))
for k in range(nbimgs):
    I = load_img('images/%03d.jpg'%k,target_size=(224,224))
    I = img_to_array(I)
    I = np.expand_dims(I,axis=0)
    I = preprocess_input(I)
    X = model_feat.predict(I)
    allfeat[k] = X.squeeze()

#%%
from skimage import io
def plot_imgs_trouvees(img,closest):
    assert np.size(closest) == 10
    
    fig,axes = plt.subplots(3,5)
    axes[0,2].imshow(img)
    for idx,i in enumerate(closest.squeeze()):
        axes[1+idx//5,idx%5].imshow(io.imread('images/%03d.jpg'%i))
        axes[1+idx//5,idx%5].set_title('{}e plus proche'.format(idx+1))
    fig.show()
  
    
from scipy.spatial import distance_matrix

cou_feat = model_feat.predict(img).squeeze()
cra_feat = model_feat.predict(img2).squeeze()
kan_feat = model_feat.predict(img3).squeeze()

#calcul des distances
cou_dists = distance_matrix(np.expand_dims(cou_feat,axis=0),allfeat,p=2).squeeze()
cra_dists = distance_matrix(np.expand_dims(cra_feat,axis=0),allfeat,p=2).squeeze()
kan_dists = distance_matrix(np.expand_dims(kan_feat,axis=0),allfeat,p=2).squeeze()

#recher des 10 plus petites
cou_10_closest = np.argsort(cou_dists)[:10]
cra_10_closest = np.argsort(cra_dists)[:10]
kan_10_closest = np.argsort(kan_dists)[:10]

#afficher
plot_imgs_trouvees(cougar,cou_10_closest)
plot_imgs_trouvees(crab,cra_10_closest)
plot_imgs_trouvees(kangaroo,kan_10_closest)


#%%
from tensorflow.keras.applications.resnet import ResNet50
model = ResNet50(weights='imagenet')
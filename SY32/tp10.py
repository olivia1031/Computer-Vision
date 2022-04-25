# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:33:41 2021

@author: ooo
"""

import numpy as np

results_train = np.loadtxt('results_train_500.txt')

label_train = np.loadtxt('label_train.txt',skiprows=638)

#%% Ex1

def iou(b1,b2):
    #deux boites avec leurs coordonnes et dimensions
    (i1,j1,h1,l1) = b1
    ii1 = i1+h1
    jj1 = j1+l1
    (i2,j2,h2,l2) = b2
    ii2 = i2+h2
    jj2 = j2+l2
    
    (i3, j3, ii3, jj3) = (max(i1,i2), max(i2,j2), min(ii1,ii2), min(jj1,jj2))
    inter = max(ii3-i3,0)*max(jj3-j3,0)
    union = (ii1-i1)*(jj1-j1) + (ii2-i2)*(jj2-j2) - inter
    
    return inter/union

#%% Ex2 
# non-maxima Suppression
def filtre_doublons(results_in, tresh_iou = 0.5):
    results_out= []
    unique_ids = np.unique(results_in[:,0])
    #unique supprimer les doublons
    
    for i in unique_ids: #image par image
        results_in_i = results_in[results_in[:,0]==i]
        #trier les boites par score de confiance croissant
        results_in_i = results_in_i.tolist()
        results_in_i.sort(key=lambda vals:vals[5],reverse=False)
        
        results_out_i=[]
        
        for n in range(len(results_in_i)-1):
            keepit =True
            for m in range(n+1,len(results_in_i)):
                if iou(results_in_i[n][1:5],results_in_i[m][1:5])>tresh_iou:
                    keepit = False
            if keepit:
                results_out_i.append(results_in_i[n])
        results_out_i.append(results_in_i[-1]) #oublier pas la dernier boite
        #
        #reorganiser les resultats, cette fois par score decroissant
        results_out_i.sort(key=lambda vals:vals[5], reverse=True)
        # ajouter les boites de cette image a la liste de toutes les boites
        results_out += results_out_i
        
    return np.array(results_out)

#%% 

results_train_filtre = filtre_doublons(results_train)

#%% Ex3 et 4

def compare(results_in_filtre, labels_in, tresh_iou = 0.5):
    #initialiser la liste resultat avec tout en FP
    vpfp = np.full((results_in_filtre.shape[0]),False)
    #idem pour une liste des visages qui ont ete ou on detectes
    detectes = np.full((labels_in.shape[0]),False)
    
    # boucle sur les visages des annotations
    # pour chaque visage, regarder si une detection va bien
    unique_ids = np.unique(results_in_filtre[:,0])
    for i in unique_ids:
        labels_in_i = labels_in[labels_in[:,0]==i]
        detectes_in_i = detectes[labels_in[:,0]==i]
        results_in_i = results_in_filtre[results_in_filtre[:,0]==i]
        vpfp_in_i = vpfp[results_in_filtre[:,0] ==i]
        
        for n in range(labels_in_i.shape[0]):
            found =False
            m=0
            while((not found) and m<results_in_i.shape[0]):
                if((not vpfp_in_i[m]) and iou(labels_in_i[n][1:5],results_in_i[m][1:5])>tresh_iou):
                    #results_in_i a partir de max score (reverse)
                    vpfp_in_i[m] = True
                    detectes_in_i[n] = True
                    found =True
                m += 1
            
        vpfp[results_in_filtre[:,0] == i] = vpfp_in_i
        detectes[labels_in[:,0] == i] = detectes_in_i
        
    return vpfp,detectes

#%%
vpfp,detectes = compare(results_train_filtre, label_train)

#%% Ex5

vp = np.count_nonzero(vpfp) #ausii =np.count_nonzero(detectes)
fp = np.count_nonzero(~vpfp)

fn = np.count_nonzero(~detectes)

#precision
prec = vp / (vp+fp)

#recall
rapp = vp / (vp+fn)

# score F1
f1 = 2*(prec*rapp) / (prec+rapp)

#%% Ex6

import matplotlib.pyplot as plt
%matplotlib auto

def plotpvr(results_filtre, labels, thresh_iou=0.5):
    #faire varier un seuil sur le score des predictions
    #liste_seuils = np.unique(results_filtre[:,5]
    liste_seuils = list(set(results_filtre[:,5].tolist())) #enlever les doublons
    liste_seuils.sort(reverse=True)
    print(liste_seuils)
    
    liste_prec = []
    liste_rapp = []
    
    for score_thresh in liste_seuils:
        results_filtre_tmp = results_filtre[results_filtre[:,5]>=score_thresh]
        vpfp,detectes = compare(results_filtre_tmp,labels,thresh_iou)
        
        vp = np.count_nonzero(vpfp)
        fp = np.count_nonzero(~vpfp)
        fn = np.count_nonzero(~detectes)
        
        #precision
        prec = vp / (vp+fp)
        
        #recall
        rapp = vp / (vp+fn)
        
        liste_prec.append(prec)
        liste_rapp.append(rapp)
    print(liste_prec)
    print(liste_rapp)
    plt.plot(liste_rapp,liste_prec)
    plt.xlabel('Rappel')
    plt.ylabel('Precision')
    plt.title('Courbe de precision/rappel')
    plt.show()
    
    return np.mean(np.array(liste_prec))

#%%

mean_prec = plotpvr(results_train_filtre,label_train)
print(mean_prec)


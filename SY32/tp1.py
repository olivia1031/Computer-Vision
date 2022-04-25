# -*- coding: utf-8 -*-
'''
counter = 100 
miles = 1000.0 
name = "John" 
print (counter)

s = 'abcdef'
print(s[0:5]+" TEST")

list = [ 'runoob', 786 , 2.23, 'john', 70.2 ]
print (list[0] )

if True:
    print ("True")
else:
    print ("False")
'''    

import matplotlib.pyplot as plt
from skimage import data

#%matplotlib auto

Ivis=data.chelsea()
Isecret = data.page( )

print(Ivis.shape)
print(Ivis.dtype)
print(Ivis.min(),Ivis.max())

print ( Isecret.shape )
print ( Isecret.dtype )
print ( Isecret.min( ),Isecret.max( ) )

fig,axes= plt.subplots(1,2)
axes[0].imshow(Ivis)
axes[0].set_title( "Chelsea " )
axes[1].imshow(Isecret,cmap='gray')
axes[1].set_title( "Page" )
plt.show( )

#%%
IvisR,IvisG,IvisB=Ivis[:,:,0],Ivis[:,:,1],Ivis[:,:,2]

fig,axes=plt.subplots(1,3)
axes[0].imshow(IvisR,cmap='gray')
axes[0].set_title("R")
axes[1].imshow(IvisG,cmap='gray')
axes[1].set_title("G")
axes[2].imshow(IvisB,cmap='gray')
axes[2].set_title("B")
plt.show()

#%% EX4
import numpy as np

Isecret2=np.full(Ivis.shape[:2],255,dtype=Isecret.dtype)
c0=(Ivis.shape[0]-Isecret.shape[0])//2
r0=(Ivis.shape[1]-Isecret.shape[1])//2
Isecret2[c0:c0+Isecret.shape[0],r0:r0+Isecret.shape[1]]=Isecret
plt.figure()
plt.imshow(Isecret2,cmap='gray')
plt.show()

#%% EX5
IsecretInv=255-Isecret2
plt.figure()
plt.imshow(IsecretInv,cmap='gray')
plt.show()

#%% EX6
maxR=np.maximum(IvisR,IsecretInv)
plt.figure()
plt.imshow(maxR,cmap='gray')
plt.show()

#%% EX7
Ifusion=np.stack([maxR,IvisG,IvisB],axis=-1)
fig,axes=plt.subplots(1,2)
axes[0].imshow(Ivis)
axes[0].set_title("Original")
axes[1].imshow(Ifusion)
axes[1].set_title("Fusion")
plt.show()




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

#%% Ex2
IvisR,IvisG,IvisB=Ivis[:,:,0],Ivis[:,:,1],Ivis[:,:,2]
fig,axes=plt.subplots(1,3)
axes[0].imshow(IvisR,cmap='gray')
axes[0].set_title("R")
axes[1].imshow(IvisG,cmap='gray')
axes[1].set_title("G")
axes[2].imshow(IvisB,cmap='gray')
axes[2].set_title("B")
plt.show()

#%% Ex4
import numpy as np

Isecret2=np.full(Ivis.shape[:2],255,dtype=Isecret.dtype)
c0=(Ivis.shape[0]-Isecret.shape[0])//2
r0=(Ivis.shape[1]-Isecret.shape[1])//2
Isecret2[c0:c0+Isecret.shape[0],r0:r0+Isecret.shape[1]]=Isecret
plt.figure()
plt.imshow(Isecret2,cmap='gray')
plt.show()

#%% Ex5
IsecretInv=255-Isecret2
plt.figure()
plt.imshow(IsecretInv,cmap='gray')
plt.show()

#%% Ex6
maxR=np.maximum(IvisR,IsecretInv)
plt.figure()
plt.imshow(maxR,cmap='gray')
plt.show()

#%% Ex7
Ifusion=np.stack([maxR,IvisG,IvisB],axis=-1)
fig,axes=plt.subplots(1,2)
axes[0].imshow(Ivis)
axes[0].set_title("Original")
axes[1].imshow(Ifusion)
axes[1].set_title("Fusion")
plt.show()

#%% Ex8
# technique LSB
truncR=IvisR & int('11110000',2) #base as 2
print(truncR.min(),truncR.max())
truncSecret=Isecret2&int('11110000',2)
print(truncSecret.min(),truncSecret.max())
fig,axes=plt.subplots(2,2)
axes[0,0].imshow(IvisR,cmap='gray')
axes[0,0].set_title("IvisR")
axes[1,0].imshow(Isecret2,cmap='gray')
axes[1,0].set_title("Isecret2")
axes[0,1].imshow(truncR,cmap='gray')
axes[0,1].set_title("truncR")
axes[1,1].imshow(truncSecret,cmap='gray')
axes[1,1].set_title("truncSecret")
plt.show()

#%% Ex9
truncSecret2=truncSecret>>4
print(truncSecret2.min(),truncSecret2.max())
lsbR=truncR|truncSecret2
print(lsbR.min(),lsbR.max())
fig,axes=plt.subplots(1,4)
axes[0].imshow(truncR,cmap='gray')
axes[0].set_title("truncR")
axes[1].imshow(truncSecret,cmap='gray')
axes[1].set_title("truncSecret")
axes[2].imshow(truncSecret2,cmap='gray')
axes[2].set_title("truncSecret2")
axes[3].imshow(lsbR,cmap='gray')
axes[3].set_title("lsbR")
plt.show()

#%% Ex10
Ifusion2=np.stack([lsbR,IvisG,IvisB],axis=-1)
fig,axes=plt.subplots(1,2)
axes[0].imshow(Ivis)
axes[0].set_title("Original")
axes[1].imshow(Ifusion2)
axes[1].set_title("Fusion LSB")
plt.show()

#%% Ex11
Ifusion2R=Ifusion2[:,:,0]
IfusionPage1=Ifusion2R & int('00001111',2)
IfusionPage2=IfusionPage1<<4
#recup=Ifusion2[:,:,0]<<4
fig,axes=plt.subplots(1,3)
axes[0].imshow(Isecret,cmap='gray')
axes[0].set_title("Original Page")
axes[1].imshow(IfusionPage1,cmap='gray')
axes[1].set_title("Recuperee Page")
axes[2].imshow(IfusionPage2,cmap='gray')
axes[2].set_title("Decaler Recuperee Page")
plt.show()

#%% Ex12
IfusionR=Ifusion2

truncG=IvisG & int('11110000',2)
lsbG=truncG|truncSecret2
IfusionG=np.stack([IvisR,lsbG,IvisB],axis=-1)

truncB=IvisB & int('11110000',2)
lsbB=truncB|truncSecret2
IfusionB=np.stack([IvisR,IvisG,lsbB],axis=-1)

fig,axes=plt.subplots(2,2)
axes[0,0].imshow(Ivis,cmap='gray')
axes[0,0].set_title("Ivis")
axes[0,1].imshow(IfusionR,cmap='gray')
axes[0,1].set_title("IfusionR")
axes[1,0].imshow(IfusionG,cmap='gray')
axes[1,0].set_title("IfusionG")
axes[1,1].imshow(IfusionB,cmap='gray')
axes[1,1].set_title("IfusionB")
plt.show()

#%% Ex13

Ivis=data.chelsea()
Isecret = data.astronaut( )
print(Ivis.shape)
print(Ivis.dtype)
print(Ivis.min(),Ivis.max())
print(Isecret.shape)
print(Isecret.dtype)
print(Isecret.min( ),Isecret.max( ) )
Isecret = Isecret[:Ivis.shape[0],:Ivis.shape[1],:]
print(Isecret.shape)
# =============================================================================
# fig,axes=plt.subplots(1,2)
# axes[0].imshow(Ivis,cmap='gray')
# axes[1].imshow(Isecret,cmap='gray')
# plt.show()
# =============================================================================

truncIvis=Ivis&int('11110000',2)
truncIsec=Isecret&int('11110000',2)
truncIsec2=truncIsec>>4
lsbRGB=truncIvis|truncIsec2

#decodage
recup=lsbRGB<<4

fig,axes=plt.subplots(2,2)
axes[0,0].imshow(Ivis,cmap='gray')
axes[0,0].set_title("Ivis")
axes[0,1].imshow(lsbRGB,cmap='gray')
axes[0,1].set_title("Fusion LSB")
axes[1,0].imshow(recup,cmap='gray')
axes[1,0].set_title("Image cachee extraite")
axes[1,1].imshow(Isecret,cmap='gray')
axes[1,1].set_title("Image cachee original")
plt.show()
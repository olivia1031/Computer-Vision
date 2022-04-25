# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from scipy.ndimage import filters
%matplotlib auto

im = camera( ).astype('float')

fig,axes= plt.subplots(1,1)
axes.imshow(im,cmap='gray')
plt.show( )
#%% Ex1.Q2

h = np.full((3,3),1/9) 
h3 = np.full((5,5),1/25) 
h4 = np.full((7,1),1/7) 
# h : filter moyen/ passe-bas/ reduit bruit et detail
# h3: filtre[5,5]plus flou, reduit plus haut de frequence dans spectre
# h4: filtre haut axe y, laisse les haut de frequences de x

#%% Ex1.Q3
imf = filters.convolve(im,h)
imf3 = filters.convolve(im,h3)
imf4 = filters.convolve(im,h4)

fig,axes= plt.subplots(1,2)
axes[0].imshow(im,cmap='gray')
axes[0].set_title( "avant convolution " )
axes[1].imshow(imf,cmap='gray')
axes[1].set_title( "apres convolution h" )
plt.show( )

#%% Ex1.Q4
from scipy import fftpack
F = fftpack.fft2(im)
Fshift = np.fft.fftshift(F)
spectre = np.log10(np.abs(Fshift)+1)

F2 = fftpack.fft2(imf)
Fshift2 = np.fft.fftshift(F2)
spectre2 = np.log10(np.abs(Fshift2)+1)

F3 = fftpack.fft2(imf3)
Fshift3 = np.fft.fftshift(F3)
spectre3 = np.log10(np.abs(Fshift3)+1)

F4 = fftpack.fft2(imf4)
Fshift4 = np.fft.fftshift(F4)
spectre4 = np.log10(np.abs(Fshift4)+1)

fig,axes= plt.subplots(4,2)
axes[0,0].imshow(im,cmap='gray')
axes[0,0].set_title( "origin" )
axes[0,1].imshow(spectre, cmap='jet')
axes[0,1].set_title( "TF" )
axes[1,0].imshow(imf,cmap='gray')
axes[1,0].set_title( "convolution" )
axes[1,1].imshow(spectre2, cmap='jet')
axes[1,1].set_title( "TF2" )

axes[2,0].imshow(imf3,cmap='gray')
axes[2,0].set_title( "h3" )
axes[2,1].imshow(spectre3, cmap='jet')
axes[2,1].set_title( "TF3" )
axes[3,0].imshow(imf4,cmap='gray')
axes[3,0].set_title( "h4" )
axes[3,1].imshow(spectre4, cmap='jet')
axes[3,1].set_title( "TF4" )

plt.show( )
#filtre elimine haut de frequence

#%% Ex2.Q1 
#Détection de contours / Filtre passe haut
im = camera( ).astype('float')

Sx = np.array([[1,0,-1],
              [2,0,-2],
              [1,0,-1]])
Sy = Sx.T

contx = filters.convolve(im,Sx)
conty = filters.convolve(im,Sy)

fig,axes= plt.subplots(1,3)
axes[0].imshow(im,cmap='gray')
axes[0].set_title( "origin" )
axes[1].imshow(contx,cmap='gray')
axes[1].set_title( "Sobel x" )
axes[2].imshow(conty,cmap='gray')
axes[2].set_title( "Sobel y" )
plt.show( )
#Sobel X detecte plus sur axe X donc contours verticaux ; contre y.

#%% Ex2.Q3
contours = np.sqrt(contx**2+conty**2)

fig,axes= plt.subplots(1,2)
axes[0].imshow(im,cmap='gray')
axes[0].set_title( "origin" )
axes[1].imshow(contours,cmap='gray')
axes[1].set_title( "Contour" )
plt.show()

#%% Ex3.Q1
from scipy.misc import ascent
im = ascent( ).astype('float')
im2 = im[0:256,0:256]
im2red = im2[::3,::3] #[start:end:step]/ a[::3] is every third element of the sequence.

fig,axes= plt.subplots(1,2)
axes[0].imshow(im2,cmap='gray')
axes[0].set_title("origin" )
axes[1].imshow(im2red,cmap='gray')
axes[1].set_title("echantillon" )
plt.show()


#%% Ex3.Q2

h = np.array([[1,8,28,56,70,56,28,8,1]])
h = h/256
H = h*h.T

conv = filters.convolve(im2,H)
convred = conv[::3,::3]

fig,axes= plt.subplots(2,2)
axes[0,0].imshow(im2,cmap='gray')
axes[0,0].set_title("origin" )
axes[0,1].imshow(im2red,cmap='gray')
axes[0,1].set_title("echantillon" )

axes[1,0].imshow(conv,cmap='gray')
axes[1,0].set_title("conv" )
axes[1,1].imshow(convred,cmap='gray')
axes[1,1].set_title("convred" )
plt.show()


#%% Ex4.Q1
N = 128
t = np.arange(0,N)
X, Y = np.meshgrid(t,t)
# =============================================================================
# print("X="+ str(X))
# print("Y="+ str(Y))
# plt.plot(X,Y, marker='.', color='k', linestyle='none')
# =============================================================================
#Return coordinate matrices from coordinate vectors.
I = 50*np.sin(2*np.pi*0.04*X+2*np.pi*0.1*Y )

from scipy import fftpack

F = fftpack.fft2(I)
Fshift = np.fft.fftshift(F)
spectre = np.log10(np.abs(Fshift)+1)

fig,axes= plt.subplots(1,2)
axes[0].imshow(I,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(spectre,cmap='jet',origin='lower',extent=[-.5,.5,-.5,.5])
axes[1].set_title("TF" )
plt.show()

#%% Ex4.Q3
N = 128
#N = 512
t = np.arange (0 ,N)
X, Y = np.meshgrid ( t , t )
I = 50*np.sin( 2*np.pi*0.04*X + 2*np.pi*0.1*Y ) + \
    50*np.sin(2*np.pi*0.06*X + 2*np.pi*0.1*Y ) + \
    0.01*np.sin(2*np.pi*0.32*X + 2*np.pi*0.2*Y )
    
from scipy import fftpack

F = fftpack.fft2(I)
Fshift = np.fft.fftshift(F)
spectre = np.log10(np.abs(Fshift)+1)

fig,axes= plt.subplots(1,2)
axes[0].imshow(I,cmap='gray')
axes[0].set_title("I" )
axes[1].imshow(spectre,cmap='jet',origin='lower',extent=[-.5,.5,-.5,.5])
axes[1].set_title("TF" )
plt.show()

#%% Ex4.Q6
from scipy.signal import kaiser
N = 128
b = 10
W = np.expand_dims(kaiser(N,b) , axis=-1) @np.expand_dims(kaiser(N,b),axis=0)

fig,axes= plt.subplots(1,2)
axes[0].plot(np.arange(N),kaiser(N,b))
axes[0].set_title("Kaiser 2D" )
axes[1].imshow(W,cmap='gray')
axes[1].set_title("Kaiser 3D" )
plt.show()

#%% Ex4.Q7
N = 128
t = np.arange (0 ,N)
X, Y = np.meshgrid ( t , t )
I = 50*np.sin( 2*np.pi*0.04*X + 2*np.pi*0.1*Y ) + \
    50*np.sin(2*np.pi*0.06*X + 2*np.pi*0.1*Y ) + \
    0.01*np.sin(2*np.pi*0.32*X + 2*np.pi*0.2*Y )
N = 128
b = 10
W = np.expand_dims(kaiser(N,b) , axis=-1) @np.expand_dims(kaiser(N,b),axis=0)
I2=I*W

from scipy import fftpack
F2 = fftpack.fft2(I2)
Fshift2 = np.fft.fftshift(F2)
spectre2 = np.log10(np.abs(Fshift2)+1)

#b petit, moins fort
#b grand, plus fin, spectre mieux
fig,axes= plt.subplots(1,2)
axes[0].imshow(spectre2,cmap='jet',origin='lower',extent=[-.5,.5,-.5,.5])
axes[0].set_title("TF" )
axes[1].imshow(W,cmap='gray')
axes[1].set_title("Kaiser 3D" )
plt.show()

#%% Ex4.Q8

from scipy.special import iv
M = np.arange(-64,64)
X, Y = np.meshgrid(M, M)
R = (X**2+Y**2 )
Rf = 64**2
b = 10
Wc = iv(0,b*np.sqrt(np.maximum(0,1-R/Rf)))/iv(0,b)
I3=I*Wc

F3 = fftpack.fft2(I3)
Fshift3 = np.fft.fftshift(F3)
spectre3 = np.log10(np.abs(Fshift3)+1)

fig,axes= plt.subplots(3,2)
axes[0,0].imshow(I,cmap='gray')
axes[0,0].set_title("I" )
axes[0,1].imshow(spectre,cmap='jet',origin='lower',extent=[-.5,.5,-.5,.5])
axes[0,1].set_title("TF" )
axes[1,0].imshow(I2,cmap='gray')
axes[1,0].set_title("I2" )
axes[1,1].imshow(spectre2,cmap='jet',origin='lower',extent=[-.5,.5,-.5,.5])
axes[1,1].set_title("TF2" )
axes[2,0].imshow(I3,cmap='gray')
axes[2,0].set_title("I3" )
axes[2,1].imshow(spectre3,cmap='jet',origin='lower',extent=[-.5,.5,-.5,.5])
axes[2,1].set_title("TF3" )
plt.show()


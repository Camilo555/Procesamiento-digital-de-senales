# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:37:29 2022

@author: Sol
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import soundfile as sf
import IPython
from scipy.linalg import svd

#%%
# señalvoz, fs = sf.read('señalvozhablada.wav')
# IPython.display.Audio("señalvozhablada.wav")


# #HAGO VENTANAS DE 30 MS
# M= 30
# #Con un oversample de 10 mseg
# O = 10
# Mm = int(M*fs/1000) #Longitud del cuadro en muestras
# Om = int(O*fs/1000) # Longitud del Oversample en muestras
# paso =int( (M-O)*fs/1000) #longitud del paso. Cada cuantas muestras se calcula el STE
# #usamos ventanas hamming? oque ocomo?
# print ('Chequeo del criterio NOLA: ', sig.check_NOLA(np.hamming(Mm),Mm,Om))
# lenseñal = len(señalvoz)

# STFT = np.array([np.zeros(Mm)]) # stft de FxM
# contador = 0
# print (len(STFT))
# for i in range(0,lenseñal-Mm,paso):
#     x_cuadro = señalvoz[i:i+Mm]*np.hamming(Mm)
#     X_f = np.fft.fft(x_cuadro)
#     STFT = np.vstack([STFT,X_f])

# stft = np.delete(STFT,(0),axis=0) 
# #stft es la stft de STFT que tuve que sacarle el primer elemento que eran ceros, no podía de otra forma hacer que funcione vstack

# U, D, VT = svd(stft)
#%%
#FUNCION DE PRUEBA DE DURACION 1 SEG

fs = 44100 # Sample Rate [Hz]
L_señal = 1*fs # Señal dura 1 seg
n = np.arange(L_señal)
t = n/fs # Vector de tiempo en seg
señal = np.sin(2*np.pi*2500*t)

plt.figure()
plt.plot(t, señal)
plt.title('Señal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
#%%
# STFT
win = 'hann'
nperseg = 300
noverlap = nperseg//2 
nfft = 2000
return_onesided = True
print(sig.check_NOLA(win,nperseg,noverlap))
f1,t1, stft = sig.stft(señal, fs = fs, window=win, nperseg=nperseg, nfft=nfft, noverlap=noverlap, return_onesided=return_onesided) 

plt.subplot(1,2,1)
plt.pcolormesh(t1, f1, 20*np.log10(np.abs(stft)) + np.finfo(float).eps, shading = "gouraud")
plt.colorbar(plt.pcolormesh(t1, f1, 20*np.log10(np.abs(stft)) + np.finfo(float).eps, shading = "gouraud"), label='dB')
plt.title('Magnitud nfft=2000')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout
plt.subplot(1,2,2)
plt.pcolormesh(t1, f1, np.angle(stft), shading = "gouraud")
plt.colorbar(plt.pcolormesh(t1, f1, np.angle(stft), shading = "gouraud"))
plt.title('Fase nfft=2000')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout
#%%
# SINGULAR VALUE DECOMPOSITION

U, D, Vh = np.linalg.svd(stft,full_matrices=True, compute_uv=True, hermitian=False)

#Nota: D es un vector con los valores de la diagonal, no es una matriz diagonal.
#Para que obtener la matriz stft devuelta habría que ubicar ese vector como la diagonal de una matriz diagonal (cosa que no hice).

#%%
#Creo matriz marca de agua como la stft de un ruido random
ruido = np.random.normal(1, 3, len(t))

plt.figure()
plt.plot(t, ruido)
plt.title('Ruido (Marca de Agua)')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')

f2,t2, stft_ruido = sig.stft(ruido, fs = fs, window=win, nperseg=nperseg, nfft=nfft, noverlap=noverlap, return_onesided=return_onesided) 

plt.subplot(1,2,1)
plt.pcolormesh(t1, f1, 20*np.log10(np.abs(stft_ruido)) + np.finfo(float).eps, shading = "gouraud")
plt.colorbar(plt.pcolormesh(t1, f1, 20*np.log10(np.abs(stft_ruido)) + np.finfo(float).eps, shading = "gouraud"), label='dB')
plt.title('Magnitud nfft=2000')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout
plt.subplot(1,2,2)
plt.pcolormesh(t1, f1, np.angle(stft_ruido), shading = "gouraud")
plt.colorbar(plt.pcolormesh(t1, f1, np.angle(stft_ruido), shading = "gouraud"))
plt.title('Fase nfft=2000')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout

#%%
#Creo matriz diagonal de FxM con el array D
D_matrix = np.empty(shape=[len(stft), len(stft[0])],dtype=complex)
np.fill_diagonal(D_matrix, D)


#Cuenta de wd (sin a ni b)
for i in range(len(stft)):
    for j in range(len(stft[0])):
        D_matrix[i,j] = D_matrix[i,j] + D_matrix[i,j]*stft_ruido[i,j]
        # D_matrix[i,j] = D[min(len(stft)-1,len(stft[0])-1)] + stft_ruido[i,j]
        


#%%
#SINGULAR VALUE DECOMPOSITION A D_matrix
Uw, Dw, Vw = np.linalg.svd(D_matrix,full_matrices=True, compute_uv=True, hermitian=False)

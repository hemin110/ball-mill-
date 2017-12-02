#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 09:05:41 2017

@author: hemin
"""

import numpy as np
from scipy.signal import welch
import scipy.io as sio
import matplotlib.pylab as plt

def pooling_mean(data_x, ventana):
    contador = 0
    limite = data_x.shape[1]%ventana
    data_final_x = np.zeros([data_x.shape[0], int(data_x.shape[1]/ventana)])
    if limite == 0:
        data_xx = np.split(data_x, data_x.shape[1]/ventana, axis=1)
    else:
        data_xx = np.split(data_x[:,:-limite], data_x.shape[1]/ventana, axis=1)
    for i in data_xx:
        data_final_x[:,contador] = i.mean(1)
        contador+=1
    return data_final_x

def main():

    fs = 51200 #sample frequency
    frame_len = int(fs/2) #hamming wondow freme lenght
    sectio_size = 131072  #sample division
    contador=0
    data_train_x = np.zeros([1529,550])  #data storage matrix
    data_test_x = np.zeros([1529,550])  #data storage matrix
    for i in range(1,140):   
        file_name = u'/media/hemin/LENOVO/数据/球磨机/处理的数据/第1次实验/'+str(i)+'_ch6.mat'
        file_data = sio.loadmat((file_name)) #extract data
        train_x = file_data["data6"]
        audio = train_x.flatten()-train_x.flatten().mean() #rest the mean
        audio = np.split(audio, range(sectio_size, audio.shape[0],sectio_size))[:22]
        data = welch(x = audio, fs= fs/2, window=np.hamming(frame_len), nperseg = frame_len, nfft=frame_len) #extrat the psv
        data = pooling_mean(data[1][:,:11000], ventana=20) #cutoff the freq beetween 11000 and 12800 and mean pooling 
        contador+=1
        print "Iteracion numero: "+repr(contador)
        idx = (i-1)*11
        data_train_x[idx:idx+11,:] = data[0:11,:]
        data_test_x[idx:idx+11,:] = data[11:22,:]
       
    sio.savemat('/home/hemin/workspace/data/train_x.mat',{'train_x':data_train_x})
    sio.savemat('/home/hemin/workspace/data/test_x.mat',{'test_x':data_test_x})

if __name__ == "__main__":
    main()
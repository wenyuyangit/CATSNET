#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:11:31 2022

@author: wenyu
"""
'''Evaluation'''
from sklearn.metrics import confusion_matrix
from Hypercolumns import *
from unet import Unet
# from Unet_Lib import *
import numpy as np
import os
import torch
import scipy.io
import matplotlib.pyplot as plt
import time

'''Parameters'''
ifc = 'nc'
which = 'DTM'     # 'DTM' / 'CHM'
step_path = '/step_1'
# iffilter = 'nofilter'
iffilter = 'filter49'
channels = 52  # 52
classes = 37 # 37 / 36
ifover = 'Overlap'

patch_size = 64   # 64, 128, 256

'''Modelname List'''

Modelname = 'Unet_5_CE_Overlap_64_0.01step02600'
# Modelname = 'Unet_1000_5_CE_Overlap_64_0.01step02600'
# Modelname = 'Unet_gau_1_1000_5_mylossKL_Overlap_64_0.01step02600'

# loss = np.load('/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/Weights/CHM_nc/step_1/filter49/Unet_gau_3_1000_5_mylossKL_Overlap_64_0.01step02600/training_loss.npy')

# pat
base_folder = '/media/wenyu/Elements/TomoSAR_Wenyu/1_SS_FCN/FullPol/Python'
save_path = '/Weights/'+which+'_'+ifc+ step_path+'/'+iffilter+'/'
folder_name = Modelname
weights_path = base_folder + save_path+ folder_name + "/Weights.pth"

'''Model list'''
# network = Hypercolumns(patch_size, classes)
# network = Hypercolumns(patch_size, classes)
num_blocks = 5
network = Unet(in_channels = channels,
                out_channels = classes,
                n_blocks = num_blocks,
                start_filters = 32,
                activation= 'relu',
                conv_mode = 'same')
# network = Unet_Res18(patch_size, classes)
# network = Unet_vgg11(classes)

network.load_state_dict(torch.load(weights_path))
device_num = 1
device = torch.device('cuda:%d'%(device_num) if torch.cuda.is_available() else "cpu")
print(device)


'''Load training process'''

# train_losses = np.load(base_folder+save_path+folder_name+"/training_loss.npy")
# validation_losses = np.load(base_folder+save_path+folder_name+"/validation_loss.npy")

# '''plot losses'''
# plt.figure()
# plt.plot(np.arange(len(train_losses)),train_losses,label="training loss")
# plt.plot(np.arange(len(validation_losses)),validation_losses, label="validation loss")
# plt.legend(loc = 'upper right')
# plt.xlabel("Epoches")
# plt.ylabel("Loss")
# plt.show()

'''def'''
def getOutput(sar_folder, GT_folder, output_name):
    _,_,files = next(os.walk(sar_folder))
    file_count = int(len(files)/2)
    output = {}
    output["predictions"] =[]
    output["GT"] =[]


    for i in range(1,file_count+1):
        sar_npy = np.load(sar_folder+"patch_SAR_"+ifc+"_"+str(i)+".npy" )
        GT_npy = np.load(GT_folder+"patch_"+which+"_"+str(i)+".npy" )
        sar_npy = np.expand_dims(sar_npy,0)
        preds = network(torch.from_numpy(sar_npy))
        _,pred = torch.max(preds.data,1)
        output["predictions"].append(pred.numpy())
        output["GT"].append(GT_npy)
    scipy.io.savemat(base_folder+save_path+folder_name+output_name,output)

'''Train dataset'''
sar_folder_train = '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/npdata/top_'+ str(patch_size) +'_'+iffilter+'/'+ifover+'/Train/'
GT_folder_train = '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/npdata/GT_'+  str(patch_size) +'_'+iffilter+'/'+ifover + step_path +'/Train/'

'''Test dataset'''
sar_folder_test  = '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/npdata/top_'+ str(patch_size) +'_'+iffilter+'/'+ifover +'/Test/'
GT_folder_test = '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/npdata/GT_'+ str(patch_size) +'_'+iffilter+'/'+ ifover + step_path+'/Test/'


# getOutput(sar_folder_test, GT_folder_test, "/test_output.mat")
# getOutput(sar_folder_train, GT_folder_train, "/train_output.mat")
#%%
# patch_folder = '/media/wenyu/Elements/TomoSAR_Wenyu/1_SS_FCN/TGRS_Unet_revise/'
# sar_npy = scipy.io.loadmat(patch_folder+"unet_Radar1_FB.mat")['Radar1'].astype(np.float32)
# # lidar = np.load(patch_folder+"test_"+which+"_1.npy")

# output = {}
# output["predictions"] =[]
# # output["GT"] =[]
# sar_npy = np.expand_dims(sar_npy,0)
# preds = network(torch.from_numpy(sar_npy))
# _,pred = torch.max(preds.data,1)
# output["predictions"].append(pred.numpy())
# scipy.io.savemat(base_folder+save_path+folder_name+"/DTM_whole_1.mat",output)

#%% Load testpatch512 data
# start_time = time.time()
patch_folder = '/media/wenyu/Elements/TomoSAR_Wenyu/1_SS_FCN/FullPol/Python/npdata/testpatch512/'
sar_npy = np.load(patch_folder+"radar_nc.npy")
lidar = np.load(patch_folder+"test_"+which+"_1.npy")

output = {}
output["predictions"] =[]
output["GT"] =[]
sar_npy = np.expand_dims(sar_npy,0)
preds = network(torch.from_numpy(sar_npy))
# end_time = time.time()
_,pred = torch.max(preds.data,1)
output["predictions"].append(pred.numpy())
output["GT"].append(lidar)
scipy.io.savemat(base_folder+save_path+folder_name+"/testpatch512.mat",output)



# elapsed_time = end_time - start_time




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:44:57 2022

@author: wenyu
"""
import pickle
from Hypercolumns import *
from unet import Unet
# from Unet_Lib import *
from diceLoss_ori import *
import torch
import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
import os
import torch.onnx
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import tensorflow as tf
from keras.utils.np_utils import *
from IoU_Jaccard import *

'''Parameters'''  # convert .mat to numpy
ifc = 'nc'
which = 'CHM' # for CHM, classes = 40
step_path = '/step_1'

batch_size = 16
iffilter = 'filter49'
channels = 52
classes = 36
ifover = 'Overlap'
# iffilter = 'nofilter'
# channels = 36 

sf = 32
patch_size = 64
preload = 0   # 1: yes, preload; 0: train the network from the scrach
loaddata = 1 # 1: yes, load the saved data;  0: reconstruct the dataset
lr = 0.01
number_epochs = 1000

base_folder = '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python'
save_path = '/Weights/'+which+'_'+ifc+ step_path+'/'+iffilter 

'''Structure'''
num_blocks = 5
network = Unet(in_channels = channels,
                out_channels = classes,
                n_blocks = num_blocks,
                start_filters = sf,
                activation= 'relu',
                conv_mode = 'same')

'''which loss'''
whichloss = 'CE'

'''pretrained Model'''
Modelname = 'Unet_'+ str(number_epochs) +'_' + str(num_blocks) +'_'+ whichloss+'_'+ifover
if (preload == 1):
    # load
    Modelname = 'Unet_5_CE_Overlap_64'
    # path
    base_folder = '/mnt/nas151/sar/TomoSAR/SS/FullPol/Python'
    save_path = '/Weights/'+which+'_'+ifc+ step_path+'/'+iffilter+'/'
    folder_name = '/'+Modelname
    weights_path = base_folder + save_path+ folder_name + "/Weights.pth"
    network.load_state_dict(torch.load(weights_path))
    
# test
dummy_sample = torch.zeros(1,channels,patch_size,patch_size)
output = network(dummy_sample)
print(f'Out:{output.shape}')

'''save the structure'''
folder_name = '/'+ Modelname +'_' + str(patch_size) + '_'+str(lr) +'step02600'
isExist = os.path.exists(base_folder+save_path+folder_name)
if not isExist:
    os.makedirs(base_folder+save_path+folder_name)
    print("Yeah! The new directory is created!")
torch.onnx.export(network,dummy_sample,base_folder+save_path+folder_name+"/Structure.onnx",input_names = ['sar'],output_names = ['height'],opset_version=11)

'''Load data'''
   
sar_folder = '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/npdata/top_64_'+iffilter+'/' + ifover+'/' +'/Train/'
GT_folder =  '/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Python/npdata/GT_64_'+ iffilter +'/' + ifover+'/' + step_path+'/Train/'

_,_,files = next(os.walk(sar_folder))
file_count = int(len(files)/2)
arr = np.arange(1,file_count+1)
np.random.shuffle(arr)
n_train = int(file_count*0.8)
n_val =  file_count - n_train

split_imgs={}
split_imgs["train"] = arr[0:n_train]
split_imgs["val"] = arr[n_train:file_count]

def to_categorical_3D(npy):
    npy_3D = np.zeros([classes, patch_size,patch_size])
    for i in range(patch_size):
        for j in range(patch_size):
            npy_3D[:,i,j] = to_categorical(npy[i,j],classes)
    return npy_3D
            
class VaihingenDataset(Dataset):
    def __init__(self, split):   
        self.sars = []
        self.GTs = []
        for i in split_imgs[split]:
            sar_npy = np.load(sar_folder+ 'patch_SAR_'+ifc+'_'+str(i) +'.npy')   # (36,64,64)
            GT_npy = np.load(GT_folder+'patch_' +which+'_'+str(i)+'.npy')   #(64,64)
            GT_npy = to_categorical_3D(GT_npy)
            self.sars.append(torch.from_numpy(sar_npy))
            self.GTs.append(torch.from_numpy(GT_npy))
    
    def __len__(self):
        return len(self.sars)
    
    def __getitem__(self, idx):
        sar = self.sars[idx]
        GT = self.GTs[idx]
        return sar, GT
       
if (loaddata == 0):         
    training_dataset = VaihingenDataset("train")
    validate_dataset = VaihingenDataset("val")
    '''save dataset'''
    output_train = open(base_folder+save_path+"/training_dataset_"+str(patch_size)+'.pkl','wb')
    str1 = pickle.dumps(training_dataset)
    output_train.write(str1)
    output_train.close()
    
    output_val = open(base_folder+save_path+"/validate_dataset_"+str(patch_size)+'.pkl','wb')
    str2 = pickle.dumps(validate_dataset)
    output_val.write(str2)
    output_val.close()
    
else:
    training_dataset = VaihingenDataset
    validate_dataset = VaihingenDataset
    
    with open(base_folder+save_path+"/training_dataset_"+str(patch_size)+'.pkl', 'rb') as file:
        training_dataset = pickle.loads(file.read())
    with open(base_folder+save_path+"/validate_dataset_"+str(patch_size)+'.pkl', 'rb') as file:
        validate_dataset = pickle.loads(file.read())
    

train_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = batch_size, shuffle=True )
validate_loader = torch.utils.data.DataLoader(dataset = validate_dataset, batch_size = batch_size, shuffle=False )

'''Train the Model'''
if iffilter == 'filter49':
    class_weights = scipy.io.loadmat('/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Data/weights/DTM_cov_weight')["DTM_cov_weight"].astype(np.float32)
    class_weights = torch.FloatTensor(class_weights).cuda()
else:
    class_weights = scipy.io.loadmat('/mnt/nas151/sar/TomoSAR/1_SS_FCN/FullPol/Data/DTM_weight')["DTM_weight"].astype(np.float32)
    class_weights = torch.FloatTensor(class_weights).cuda()

loss_function_DICE = DiceLoss()
loss_function_CE = torch.nn.CrossEntropyLoss()
    
optimizer = torch.optim.SGD(network.parameters(),lr=lr)
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 600, gamma=0.2)

device_num = 0
device = torch.device('cuda:%d'%(device_num) if torch.cuda.is_available() else "cpu")
print(device)

network.to(device)
 
training_losses = []
training_ioues = []
training_acces = []
validation_losses1 = []
validation_losses2 = []
validation_losses = []
validation_ioues = []
validation_acces = []

bar_loss = 1e5
y = []
for epoch in range(number_epochs):
    print("Starting epoch number "+str(epoch))
    
    # Train phase:
    train_loss = 0
    train_iou = 0
    train_acc = 0
    
    network.train()
    total = 0
    for i,(inputs, GTs) in enumerate(tqdm(train_loader)):  # i = 0,1,..,len(num_trainsamples)/batch_size
        # Convert the inputs and GTs to torch Variables
        inputs = Variable(inputs).to(device)
        GTs = Variable(GTs).to(device)
        optimizer.zero_grad()
        pred = network(inputs)
        GTs_dim = torch.argmax(GTs,dim=1)
        # loss function
        # loss = 10*loss_function_DICE(pred,GTs) + loss_function_CE(pred,GTs_dim)
        loss1 = loss_function_CE(pred,GTs_dim)
        # loss2 = loss_function_DICE(pred,GTs)
        loss = loss1
        train_iou += mIOU(GTs_dim,pred,classes,eps=1e-8)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item() # I do not think there shoule be /len(train_loader)
        pred_dim = torch.argmax(pred,dim=1)
        train_acc += (pred_dim == GTs_dim).sum().item()
        total += (GTs_dim == GTs_dim).sum().item()
        
    scheduler.step()
    lr = scheduler.get_lr()
    y.append(optimizer.param_groups[0]['lr'])
    # train_iou = train_iou/len(train_loader) 
    train_loss = train_loss/len(train_loader)
    # train_acc = train_acc/total
    print("Epoch-------------" + str(epoch) +": train loss = " + str(train_loss))
    print("Epoch-------------" + str(epoch) +": train iou = " + str(train_iou))
    training_losses.append(train_loss)
    # training_ioues.append(train_iou)
    # training_acces.append(train_acc)
    
    # Validation phase
    validation_loss = 0
    validation_iou = 0
    validation_acc = 0
    validation_loss1 = 0
    validation_loss2 = 0
    
    # network.eval()
    
    total = 0
    with torch.no_grad():
        for i, (inputs, GTs) in enumerate(tqdm(validate_loader)):
            inputs = Variable(inputs).to(device)
            GTs = Variable(GTs).to(device)
            pred = network(inputs)
            GTs_dim = torch.argmax(GTs,dim=1)
            # loss_function
            # loss = loss_function_DICE(pred,GTs) + loss_function_CE(pred,GTs_dim)
            loss1 = loss_function_CE(pred,GTs_dim)
            # loss2 = loss_function_DICE(pred,GTs)
            loss = loss1
            validation_iou += mIOU(GTs_dim,pred,classes,eps=1e-8)
            validation_loss += loss.cpu().item()
            # validation_loss1 += loss1.cpu().item()
            # validation_loss2 += loss2.cpu().item()
            pred_dim = torch.argmax(pred,dim=1)
            validation_acc += (pred_dim == GTs_dim).sum().item()
            total += (GTs_dim == GTs_dim).sum().item()
            

    # validation_loss1 = validation_loss1/len(validate_loader)
    # validation_loss2 = validation_loss2/len(validate_loader)      
    validation_loss = validation_loss/len(validate_loader)
    # validation_acc = validation_acc/total
    # validation_iou = validation_iou/len(validate_loader)
    print("Epoch-------------" + str(epoch) +", validation loss=" + str(validation_loss))
    
    validation_losses.append(validation_loss)    
    # validation_losses1.append(validation_loss1)
    # validation_losses2.append(validation_loss2)
    # validation_ioues.append(validation_iou)
    # validation_acces.append(validation_acc)
    
    
    if validation_loss < bar_loss:
        torch.save(network.state_dict(),base_folder+save_path+folder_name+"/Weights.pth")    
        bar_loss = validation_loss 
        print("Epoch-------------" + str(epoch) +" save network, loss = " + str(bar_loss))
        
        
    np.save(base_folder+save_path+folder_name+"/training_loss.npy",training_losses)
    # np.save(base_folder+save_path+folder_name+"/training_iou.npy",training_ioues)
    # np.save(base_folder+save_path+folder_name+"/training_acc.npy",training_acces)
    np.save(base_folder+save_path+folder_name+"/validation_loss.npy",validation_losses)
    # np.save(base_folder+save_path+folder_name+"/validation_loss1.npy",validation_losses1)
    # np.save(base_folder+save_path+folder_name+"/validation_loss2.npy",validation_losses2)
    # np.save(base_folder+save_path+folder_name+"/validation_iou.npy",validation_ioues)
    # np.save(base_folder+save_path+folder_name+"/validation_acc.npy",validation_acces)
    np.save(base_folder+save_path+folder_name+"/lr.npy",y)






'''
CGAN training/tuning script for environmental predictors (MSLP, T2, RH, etc)
'''

# general tools
import os
import sys
from glob import glob

# data tools
import time
import h5py
import random
import numpy as np
from random import shuffle
from tensorflow import keras
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

var_names=[
    'Max/Comp Radar', 
    'MSLP',
    'UH 2-5 km',
    'UH 0-2 km',
    'Graupel mass',
    'T 2m',
    'Dewpoint 2m',
    'SPD 10m',
    'APCP',
    'CAPE',
    'CIN',
    'SRH 0-3 km',
    'SRH 0-1 km',
    'U shear 0-6 km',
    'V shear 0-6 km']

input_flag = [False, False, True, True, True, False, False, True, True, False, False, False, False, False, False]
output_skew_flag = [True, False, False, False, False, False, False, False, False, True, True, False, False, False, False]
output_norm_flag = [False, True, False, False, False, True, True, False, False, False, False, True, True, True, True]
# Radar MSLP, T2, Dewpoint, CAPE, CIN, SRH3, SRH1, U Shear, V Shear

print('conditional input: {}'.format(list(np.array(var_names)[input_flag])))
print('GAN output: {}'.format(list(np.array(var_names)[output_skew_flag])))
print('GAN output: {}'.format(list(np.array(var_names)[output_norm_flag])))

weights_round = 4
save_round = 5
seeds = 123456
model_de_load = 'GAN_de{}_norm_wild'.format(weights_round) #False
model_gen_load = 'GAN_gen{}_norm_wild'.format(weights_round) #False

model_de_save = 'GAN_de{}_norm_wild'.format(save_round) #False
model_gen_save = 'GAN_gen{}_norm_wild'.format(save_round) #False

lr = 1e-4
loss_weight = 1.0
std_noise = 0.5
non_neg = False
output_flag = output_norm_flag

N_envir = sum(output_flag)
N_micro = sum(input_flag)

vers = ['v3', 'v4x', 'v4']
leads = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

filenames_pos_train = np.load(save_dir_campaign+'HRRR_filenames_pos.npy', allow_pickle=True)[()]
filenames_neg_train = np.load(save_dir_campaign+'HRRR_filenames_neg.npy', allow_pickle=True)[()]

# ------------------------------------------------------------------ #
# Merge train/valid and pos/neg batch files from multiple lead times
pos_train_all = []
neg_train_all = []

for ver in vers:
    for lead in leads:
        pos_train_all += filenames_pos_train['{}_lead{}'.format(ver, lead)]
        neg_train_all += filenames_neg_train['{}_lead{}'.format(ver, lead)]
        
pos_train_v3 = []
neg_train_v3 = []

pos_train_v4x = []
neg_train_v4x = []

pos_train_v4 = []
neg_train_v4 = []

for lead in leads:
    pos_train_v3 += filenames_pos_train['{}_lead{}'.format('v3', lead)]
    neg_train_v3 += filenames_neg_train['{}_lead{}'.format('v3', lead)]
    
    pos_train_v4x += filenames_pos_train['{}_lead{}'.format('v4x', lead)]
    neg_train_v4x += filenames_neg_train['{}_lead{}'.format('v4x', lead)]
    
    pos_train_v4 += filenames_pos_train['{}_lead{}'.format('v4', lead)]
    neg_train_v4 += filenames_neg_train['{}_lead{}'.format('v4', lead)]
    
filenames_train = pos_train_v3 + neg_train_v3 + pos_train_v4x + neg_train_v4x

class gan_sample_gen(keras.utils.Sequence):
    def __init__(self, filename, input_flag, output_flag, std_noise, non_neg):
        '''
        Data generator class for Keras models
            - filename: list of numpy files, 1 file = 1 batch, 1 list = 1 epoch
            - labels: e.g., ['input_label', 'target_label']
            - input_flag: flag of channels, channel last. e.g., [True, False, True]
            - output_flag: _
        '''
        self.filename = filename
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.std_noise = std_noise
        self.non_neg = non_neg
        self.filenum = len(self.filename)
        
    def __len__(self):
        return self.filenum
    
    def __getitem__(self, index):
        temp_name = self.filename[index]
        return self.__readfile__(temp_name, self.input_flag, self.output_flag, self.std_noise, self.non_neg)
    
    def __readfile__(self, temp_name, input_flag, output_flag, std_noise, non_neg):
        
        N_envir = sum(output_flag)
        N_micro = sum(input_flag)
        
        data_temp = np.load(temp_name, allow_pickle=True)
        
        X = data_temp[..., input_flag]
        Y = data_temp[..., output_flag]
        
        noise = np.random.normal(0, std_noise, size=(1, 64, 64, N_envir))
        Y_noise = Y+noise
        
        if non_neg:
            Y_noise[Y_noise<0] = 0
            
        return [Y_noise, X], [Y]
    
filename_valid = (pos_train_v4+neg_train_v4)[::10000]
gen_valid = gan_sample_gen(filename_valid, input_flag, output_flag, std_noise, non_neg=non_neg)

model_gen = mu.GAN_G(N_envir, N_micro)
model_de = mu.GAN_D(N_envir, N_micro)

IN_envir = keras.Input((64, 64, N_envir))
IN_micro = keras.Input((64, 64, N_micro))

G_OUT = model_gen([IN_envir, IN_micro])
D_OUT = model_de([G_OUT, IN_micro])

# GAN = keras.models.Model([IN_envir, IN_micro], [D_OUT])
GAN = keras.models.Model([IN_envir, IN_micro], [G_OUT, D_OUT])

model_de.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(lr=1e-4))

model_gen.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(lr=1e-4))

# GAN.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(lr=1e-4))
GAN.compile(loss=[keras.losses.mean_absolute_error, keras.losses.BinaryCrossentropy(from_logits=False)], 
            loss_weights=[1.0, loss_weight], optimizer=keras.optimizers.Adam(lr=1e-4))

if weights_round > 0:
    W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_de_load))
    model_de.set_weights(W_old)
    W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_gen_load))
    model_gen.set_weights(W_old)
    
epochs = 200
L_train = 64
batch_size = 200
L = len(filenames_train)

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

# loss backup
GAN_LOSS = np.zeros([int(epochs*L_train), 3])*np.nan
D_LOSS = np.zeros([int(epochs*L_train)])*np.nan
V_LOSS = np.zeros([epochs])



X_batch = np.empty((batch_size, 64, 64, N_micro))
Y_batch = np.empty((batch_size, 64, 64, N_envir))
Y_batch_noise = np.empty((batch_size, 64, 64, N_envir))

y_bad = np.zeros(batch_size)
y_good = np.ones(batch_size)
dummy_good = y_good
dummy_mix = np.concatenate((y_bad, y_good), axis=0)

for i in range(epochs):
    print('epoch = {}'.format(i))
    if i == 0:
        record = model_gen.evaluate(gen_valid, verbose=1)
        print('Initial validation loss: {}'.format(record))
        
    start_time = time.time()
    # loop over batches
    
    for j in range(L_train):
        
        inds_rnd = du.shuffle_ind(L)
        inds_ = inds_rnd[:batch_size]
        
        for k, ind in enumerate(inds_):
            # import batch data
            temp_name = filenames_train[ind]
            data_temp = np.load(temp_name, allow_pickle=True)
            X_batch[k, ...] = data_temp[..., input_flag]
            Y_batch[k, ...] = data_temp[..., output_flag]
            
        noise = np.random.normal(0, std_noise, size=(batch_size, 64, 64, N_envir))
        Y_batch_noise = Y_batch + noise
        
        if non_neg:
            Y_batch_noise[Y_batch_noise<0] = 0
        
        # get G_output
        model_de.trainable = True
        g_out = model_gen.predict([Y_batch_noise, X_batch]) # <-- np.array

        # test D with G_output
        d_in_Y = np.concatenate((g_out, Y_batch), axis=0) # batch size doubled
        d_in_X = np.concatenate((X_batch, X_batch), axis=0)
        d_target = dummy_mix
        
        batch_ind = du.shuffle_ind(2*batch_size)
        d_loss = model_de.train_on_batch([d_in_Y[batch_ind, ...], d_in_X[batch_ind, ...]], 
                                          d_target[batch_ind, ...])

        # G training / transferring
        model_de.trainable = False
        gan_in = [Y_batch_noise, X_batch]
        gan_target = [Y_batch, dummy_good]
        #gan_target = [dummy_good,]

        gan_loss = GAN.train_on_batch(gan_in, gan_target)
        # # Backup training loss
        D_LOSS[i*L_train+j] = d_loss
        GAN_LOSS[i*L_train+j, :] = gan_loss
        #
        if j%10 == 0:
            print('\t{} step loss = {}'.format(j, gan_loss))
    # on epoch-end
    
    # save model regardless
    model_de.save('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_de_save))
    model_gen.save('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_gen_save))
    
    record_temp = model_gen.evaluate(gen_valid, verbose=1)
    # Backup validation loss
    V_LOSS[i] = record_temp
    
    # print out valid loss change
    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        
    save_dict = {}
    save_dict['D_LOSS'] = D_LOSS
    save_dict['GAN_LOSS'] = GAN_LOSS
    save_dict['V_LOSS'] = V_LOSS
    save_loss_name = '/glade/work/ksha/NCAR/Keras_models/LOSS_{}.npy'.format(model_gen_save)
    np.save(save_loss_name, save_dict)
    
    print(save_loss_name)
    print("--- %s seconds ---" % (time.time() - start_time))
    # mannual callbacks


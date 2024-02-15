# general tools
import os
import sys
from glob import glob

# data tools
import re
import time
import h5py
import random
import numpy as np
import scipy.ndimage
from random import shuffle
from tensorflow import keras
from datetime import datetime, timedelta

#tf.config.run_functions_eagerly(True)

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

def verif_metric(VALID_target, Y_pred):
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    metric = BS
    return metric

def neighbour_leads(lead):
    out = [lead-2, lead-1, lead, lead+1]
    flag_shift = [0, 0, 0, 0]
    
    for i in range(4):
        if out[i] < 0:
            out[i] = 24+out[i]
            flag_shift[i] = -1
        if out[i] > 23:
            out[i] = out[i]-24
            flag_shift[i] = +1
            
    return out, flag_shift


def filename_to_loc(filenames):
    lead_out = []
    indx_out = []
    indy_out = []
    day_out = []
    
    for i, name in enumerate(filenames):
        
        nums = re.findall(r'\d+', name)
        
        lead = int(nums[-1])
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
      
        indx_out.append(indx)
        indy_out.append(indy)
        day_out.append(day)
        lead_out.append(lead)
        
    return np.array(indx_out), np.array(indy_out), np.array(day_out), np.array(lead_out)

# ==================== #
seeds = 123456 # random seed

weights_round = 4
save_round = 5
model_prefix_load = 'RE2_smooth_vgg{}'.format(weights_round) #False
model_prefix_save = 'RE2_smooth_vgg{}'.format(save_round)

N_vars = L_vars = 15
lr = 5e-5 # learning rate

re_compute = False # re-compute severe weather labels for each training batch 
# ==================== #

vers = ['v3', 'v4x', 'v4'] # HRRR v4, v4x, v4
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

# compute severe weather labels for each training batch
if re_compute:   
    label_smooth_v3 = ()
    label_smooth_v4x = ()
    label_smooth_v4 = ()
    
    for lead in leads:
    
        lead_window, flag_shift = neighbour_leads(lead)
        
        print('Collect HRRR v3 labels ...')
        
        record_all = ()
    
        for i, lead_temp in enumerate(lead_window):
    
            flag_ = flag_shift[i]
    
            with h5py.File(save_dir_scratch+'SPC_to_lead{}_72km_all.hdf'.format(lead_temp), 'r') as h5io:
                record_temp = h5io['record_v3'][...]
    
            if flag_shift[i] == 0:
                record_all = record_all + (record_temp,)
    
            if flag_shift[i] == -1:
                record_temp[1:, ...] = record_temp[:-1, ...]
                record_temp[0, ...] = np.nan
                record_all = record_all + (record_temp,)
    
            if flag_shift[i] == +1:
                record_temp[:-1, ...] = record_temp[1:, ...]
                record_temp[-1, ...] = np.nan
                record_all = record_all + (record_temp,)
    
        shape_record = record_temp.shape      
        record_v3 = np.empty(shape_record)
        record_v3[...] = 0.0 #np.nan
    
        for i in range(4):
            record_temp = record_all[i]
            for day in range(shape_record[0]):
                for ix in range(shape_record[1]):
                    for iy in range(shape_record[2]):
                        for event in range(shape_record[3]):
                            if record_temp[day, ix, iy, event] > 0:
                                record_v3[day, ix, iy, event] = 1.0
                            elif record_v3[day, ix, iy, event] == 1.0:
                                record_v3[day, ix, iy, event] = 1.0
                            else:
                                record_v3[day, ix, iy, event] = 0.0
        label_smooth_v3 += (record_v3[None, ...],)
        
        print('Collect HRRR v4x labels ...')
        
        record_all = ()
    
        for i, lead_temp in enumerate(lead_window):
    
            flag_ = flag_shift[i]
    
            with h5py.File(save_dir_scratch+'SPC_to_lead{}_72km_v4x.hdf'.format(lead_temp), 'r') as h5io:
                record_temp = h5io['record_v4x'][...]
    
            if flag_shift[i] == 0:
                record_all = record_all + (record_temp,)
    
            if flag_shift[i] == -1:
                record_temp[1:, ...] = record_temp[:-1, ...]
                record_temp[0, ...] = np.nan
                record_all = record_all + (record_temp,)
    
            if flag_shift[i] == +1:
                record_temp[:-1, ...] = record_temp[1:, ...]
                record_temp[-1, ...] = np.nan
                record_all = record_all + (record_temp,)
    
    
        shape_record = record_temp.shape      
        record_v4x = np.empty(shape_record)
        record_v4x[...] = np.nan
    
        for i in range(4):
            record_temp = record_all[i]
            for day in range(shape_record[0]):
                for ix in range(shape_record[1]):
                    for iy in range(shape_record[2]):
                        for event in range(shape_record[3]):
                            if record_temp[day, ix, iy, event] > 0:
                                record_v4x[day, ix, iy, event] = 1.0
                            elif record_v4x[day, ix, iy, event] == 1.0:
                                record_v4x[day, ix, iy, event] = 1.0
                            else:
                                record_v4x[day, ix, iy, event] = 0.0
        
        label_smooth_v4x += (record_v4x[None, ...],)
        
        print('Collect HRRR v4 labels ...')
        
        record_all = ()
        
        for i, lead_temp in enumerate(lead_window):
    
            flag_ = flag_shift[i]
    
            with h5py.File(save_dir_scratch+'SPC_to_lead{}_72km_all.hdf'.format(lead_temp), 'r') as h5io:
                record_temp = h5io['record_v4'][...]
    
            if flag_shift[i] == 0:
                record_all = record_all + (record_temp,)
    
            if flag_shift[i] == -1:
                record_temp[1:, ...] = record_temp[:-1, ...]
                record_temp[0, ...] = np.nan
                record_all = record_all + (record_temp,)
    
            if flag_shift[i] == +1:
                record_temp[:-1, ...] = record_temp[1:, ...]
                record_temp[-1, ...] = np.nan
                record_all = record_all + (record_temp,)
                
                
        shape_record = record_temp.shape      
        record_v4 = np.empty(shape_record)
        record_v4[...] = 0.0 #np.nan
    
        for i in range(4):
            record_temp = record_all[i]
            for day in range(shape_record[0]):
                for ix in range(shape_record[1]):
                    for iy in range(shape_record[2]):
                        for event in range(shape_record[3]):
                            if record_temp[day, ix, iy, event] > 0:
                                record_v4[day, ix, iy, event] = 1.0
                            elif record_v4[day, ix, iy, event] == 1.0:
                                record_v4[day, ix, iy, event] = 1.0
                            else:
                                record_v4[day, ix, iy, event] = 0.0
                                
        label_smooth_v4 += (record_v4[None, ...],)
        
        print('... Done')
        
    label_concat_v3 = np.concatenate(label_smooth_v3, axis=0)
    label_concat_v4x = np.concatenate(label_smooth_v4x, axis=0)
    label_concat_v4 = np.concatenate(label_smooth_v4, axis=0)
    
    label_concat_v3 = np.sum(label_concat_v3, axis=-1)
    label_concat_v3[label_concat_v3>1] = 1
    
    label_concat_v4 = np.sum(label_concat_v4, axis=-1)
    label_concat_v4[label_concat_v4>1] = 1
    
    label_concat_v4x = np.sum(label_concat_v4x, axis=-1)
    label_concat_v4x[label_concat_v4x>1] = 1
    
    shape_label_v3 = label_concat_v3.shape
    shape_label_v4 = label_concat_v4.shape
    shape_label_v4x = label_concat_v4x.shape
    
    label_final_v3 = np.empty(shape_label_v3)
    label_final_v4 = np.empty(shape_label_v4)
    label_final_v4x = np.empty(shape_label_v4x)
    
    for i in range(shape_label_v3[0]):
        for j in range(shape_label_v3[1]):
            label_final_v3[i, j, ...] = scipy.ndimage.gaussian_filter(label_concat_v3[i, j], sigma=2.5)
            
    for i in range(shape_label_v4[0]):
        for j in range(shape_label_v4[1]):
            label_final_v4[i, j, ...] = scipy.ndimage.gaussian_filter(label_concat_v4[i, j], sigma=2.5)
            
    for i in range(shape_label_v4x[0]):
        for j in range(shape_label_v4x[1]):
            label_final_v4x[i, j, ...] = scipy.ndimage.gaussian_filter(label_concat_v4x[i, j], sigma=2.5)
            
    indx_pos_train_v3, indy_pos_train_v3, day_pos_train_v3, lead_pos_train_v3 = filename_to_loc(pos_train_v3)
    indx_neg_train_v3, indy_neg_train_v3, day_neg_train_v3, lead_neg_train_v3 = filename_to_loc(neg_train_v3)
    
    lead_pos_train_v3 = lead_pos_train_v3 - 2
    lead_neg_train_v3 = lead_neg_train_v3 - 2
    
    y_pos_train_v3 = label_final_v3[lead_pos_train_v3, day_pos_train_v3, indx_pos_train_v3, indy_pos_train_v3]
    y_neg_train_v3 = label_final_v3[lead_neg_train_v3, day_neg_train_v3, indx_neg_train_v3, indy_neg_train_v3]   
    
    indx_pos_train_v4x, indy_pos_train_v4x, day_pos_train_v4x, lead_pos_train_v4x = filename_to_loc(pos_train_v4x)
    indx_neg_train_v4x, indy_neg_train_v4x, day_neg_train_v4x, lead_neg_train_v4x = filename_to_loc(neg_train_v4x)
    
    lead_pos_train_v4x = lead_pos_train_v4x - 2
    lead_neg_train_v4x = lead_neg_train_v4x - 2
    
    y_pos_train_v4x = label_final_v4x[lead_pos_train_v4x, day_pos_train_v4x, indx_pos_train_v4x, indy_pos_train_v4x]
    y_neg_train_v4x = label_final_v4x[lead_neg_train_v4x, day_neg_train_v4x, indx_neg_train_v4x, indy_neg_train_v4x]
    
    y_pos_train_all = np.concatenate((y_pos_train_v3, y_pos_train_v4x,), axis=0)
    y_neg_train_all = np.concatenate((y_neg_train_v3, y_neg_train_v4x,), axis=0)
    
    y_pos_train_all_adjust = np.copy(y_pos_train_all) + 0.75
    y_pos_train_all_adjust[y_pos_train_all_adjust>0.99] = 0.99
    
    y_neg_train_all_adjust = np.copy(y_neg_train_all)
    # y_neg_train_all_adjust[y_neg_train_all_adjust>0.49] = 0.49
    
    np.save(save_dir_campaign+'y_pos_train_all_adjust.npy', y_pos_train_all_adjust)
    np.save(save_dir_campaign+'y_neg_train_all_adjust.npy', y_neg_train_all_adjust)

else:
    y_pos_train_all_adjust = np.load(save_dir_campaign+'y_pos_train_all_adjust.npy')
    y_neg_train_all_adjust = np.load(save_dir_campaign+'y_neg_train_all_adjust.npy')

ind_pick_from_batch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

with h5py.File(save_dir+'CNN_Validation_large.hdf', 'r') as h5io:
    VALID_input_64 = h5io['VALID_input_64'][...]
    VALID_target = h5io['VALID_target'][...]
    
    
# ----------------------------------------------------------------- #
# model and weights
model_head = mu.create_model_head(input_shape=(128,), N_node=64)
model_base = mu.create_model_vgg(input_shape=(64, 64, 15), channels=[48, 64, 96, 128])

IN = keras.layers.Input(shape=(64, 64, 15))

VEC = model_base(IN)
OUT = model_head(VEC)

model_final = keras.models.Model(inputs=IN, outputs=OUT)


# ============================================= #
# Weights
if weights_round > 0:
    if model_prefix_load:
        W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_prefix_load))
        model_final.set_weights(W_old)
    
model_final.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(lr=lr))

# ----------------------------------------------------------------- #
# model training loop
Y_pred = model_final.predict([VALID_input_64])
record_temp = verif_metric(VALID_target, Y_pred)

# training parameters
epochs = 500
L_train = 64
min_del = 0.0
max_tol = 100 # early stopping with patience
batch_size = 200

# Allocate batch files
X_batch_64 = np.empty((batch_size, 64, 64, L_vars))
Y_batch = np.empty((batch_size, 1))
X_batch_64[...] = np.nan
Y_batch[...] = np.nan

# Model check-point info
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'
model_name = model_prefix_save
model_path = temp_dir + model_name

# ========== Training loop ========== #
tol = 0 # initial tol

filename_pos_train = pos_train_all
filename_neg_train = neg_train_all
L_pos = len(filename_pos_train)
L_neg = len(filename_neg_train)

record = record_temp
print("Initial record: {}".format(record))

mu.set_seeds(seeds)
    
for i in range(epochs):
    start_time = time.time()

    # loop of batch
    for j in range(L_train):
        N_pos = 20
        N_neg = batch_size - N_pos

        ind_neg = du.shuffle_ind(L_neg)
        ind_pos = du.shuffle_ind(L_pos)
        
        # neg batches from this training rotation 
        file_pick_neg = []
        file_label_neg = []
        for ind_temp in ind_neg[:N_neg]:
            file_pick_neg.append(filename_neg_train[ind_temp])
            file_label_neg.append(y_neg_train_all_adjust[ind_temp])
            
        # pos batches from this training rotation 
        file_pick_pos = []
        file_label_pos = []
        for ind_temp in ind_pos[:N_pos]:
            file_pick_pos.append(filename_pos_train[ind_temp])
            file_label_pos.append(y_pos_train_all_adjust[ind_temp])
            
        # get all the batch filenames for checking labels
        file_pick = file_pick_neg + file_pick_pos
        file_label = file_label_neg + file_label_pos
        Y_batch = np.array(file_label)[:, None]

        # Assign labels based on batch filenames
        for k in range(batch_size):
            #print(file_pick[k])
            data = np.load(file_pick[k], allow_pickle=True)
            for l, c in enumerate(ind_pick_from_batch):
                temp = data[..., c] 
                X_batch_64[k, ..., l] = temp
                
        # ------------------------------------------------- #
        # batch input and label from this training rotation 
        ind_ = du.shuffle_ind(batch_size)
        X_batch_64 = X_batch_64[ind_, ...]
        Y_batch = Y_batch[ind_, :]

        # train on batch
        model_final.train_on_batch(X_batch_64, Y_batch);

    # epoch end operations
    Y_pred = model_final.predict([VALID_input_64])
    record_temp = verif_metric(VALID_target, Y_pred)

    if (record - record_temp > min_del):
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0
        print('save to: {}'.format(model_path))
        model_final.save(model_path)
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        if record_temp >= 2.0:
            print('Early stopping')
            break;
        else:
            tol += 1
            if tol >= max_tol:
                print('Early stopping')
                break;
            else:
                continue;
    print("--- %s seconds ---" % (time.time() - start_time))
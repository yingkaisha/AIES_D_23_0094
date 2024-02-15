'''
Train classifier head with size-128 feature vectors on a 64-by-64 grid cell
Nearby grid cells are not considered
Revamped
'''

# general tools
import os
import re
import sys
import time
import h5py
import random
import scipy.ndimage
from glob import glob

import numpy as np
from datetime import datetime, timedelta
from random import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

def feature_extract(filenames, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max):
    
    lon_out = []
    lat_out = []
    elev_out = []

    for i, name in enumerate(filenames):
        
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        
        lon = lon_80km[indx, indy]
        lat = lat_80km[indx, indy]

        lon = (lon - lon_minmax[0])/(lon_minmax[1] - lon_minmax[0])
        lat = (lat - lat_minmax[0])/(lat_minmax[1] - lat_minmax[0])

        elev = elev_80km[indx, indy]
        elev = elev / elev_max
        
        lon_out.append(lon)
        lat_out.append(lat)
        elev_out.append(elev)
        
    return np.array(lon_out), np.array(lat_out), np.array(elev_out)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============== #

lead = int(args['lead'])
lead_name = lead
model_tag = 'base' #'vgg2'

N = 50

# ================================================================ #
# Geographical information

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    elev_3km = h5io['elev_3km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    
grid_shape = land_mask_80km.shape

elev_80km = du.interp2d_wraper(lon_3km, lat_3km, elev_3km, lon_80km, lat_80km, method='linear')

elev_80km[np.isnan(elev_80km)] = 0
elev_80km[elev_80km<0] = 0
elev_max = np.max(elev_80km)

lon_80km_mask = lon_80km[land_mask_80km]
lat_80km_mask = lat_80km[land_mask_80km]

lon_minmax = [np.min(lon_80km_mask), np.max(lon_80km_mask)]
lat_minmax = [np.min(lat_80km_mask), np.max(lat_80km_mask)]

# ============================================================ #
# File path
path_name1_v3 = path_batch_v3
path_name2_v3 = path_batch_v3
path_name3_v3 = path_batch_v3
path_name4_v3 = path_batch_v3

path_name1_v4 = path_batch_v4x
path_name2_v4 = path_batch_v4x
path_name3_v4 = path_batch_v4x
path_name4_v4 = path_batch_v4x

path_name1_v4_test = path_batch_v4
path_name2_v4_test = path_batch_v4
path_name3_v4_test = path_batch_v4
path_name4_v4_test = path_batch_v4

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

# ====================================================== #
# HRRR v4x validation set
# ====================================================== #
# Read batch file names (npy)

filename_valid_lead = sorted(glob("{}TEST*lead{}.npy".format(path_name1_v4_test, lead)))

# =============================== #
# Load feature vectors

valid_lead = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead, model_tag), allow_pickle=True)[()]
VALID_lead = valid_lead['y_vector']
VALID_lead_y = valid_lead['y_true']

# ============================================================ #
# Consistency check indices

# if lead == 2:
#     lead1_ = 2
#     lead2_ = 3
#     lead3_ = 4
#     lead4_ = 5
# elif lead == 3:
#     lead1_ = 2
#     lead2_ = 3
#     lead3_ = 4
#     lead4_ = 5
# else:
#     lead1_ = lead-2
#     lead2_ = lead-1
#     lead3_ = lead
#     lead4_ = lead+1

IND_TEST_lead = np.load('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy', allow_pickle=True)[()]
VALID_ind = IND_TEST_lead['lead{}'.format(lead)]

# ================================================================== #
# Collect feature vectors from all batch files

L = len(VALID_ind)

filename_valid_pick = []
VALID_X_lead = np.empty((L, 128))
VALID_Y = np.empty(L)

for i in range(L):
    
    ind_lead = int(VALID_ind[i])
    
    filename_valid_pick.append(filename_valid_lead[ind_lead])
    VALID_X_lead[i, :] = VALID_lead[ind_lead, :]
    VALID_Y[i] = VALID_lead_y[ind_lead]
    
VALID_VEC = VALID_X_lead

# ================================================================== #
# extract location information

lon_norm, lat_norm, elev_norm = feature_extract(filename_valid_pick, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)
VALID_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None]), axis=1)

# ================================================================================ #
# Inference

def create_classif_head():
    
    IN = keras.Input((128,))
    IN_vec = keras.Input((2,))
    
    X = keras.layers.Concatenate()([IN, IN_vec])
    
    X = keras.layers.Dense(128)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=[IN, IN_vec], outputs=OUT)
    return model

model = create_classif_head()
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(lr=0))

Y_pred_test_ens = np.empty((len(VALID_Y), N)); Y_pred_test_ens[...] = np.nan
# Y_pred_train_ens = np.empty((len(ALL_VEC), N)); Y_pred_train_ens[...] = np.nan

valid_shape = VALID_VEC.shape
#train_shape = ALL_VEC.shape

for n in range(N):

    print("round {}".format(n))

    W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}_lead{}_base{}'.format(model_tag, lead_name, n))
    model.set_weights(W_old)
    
    Y_pred_test_ens[:, n] = model.predict([VALID_VEC, VALID_stn])[:, 0]
    #Y_pred_train_ens[:, n] = model.predict([ALL_VEC, ALL_stn])[:, 0]

# Save results
save_dict = {}
save_dict['Y_pred_test_ens'] = Y_pred_test_ens
save_dict['VALID_Y'] = VALID_Y

#save_dict['Y_pred_train_ens'] = Y_pred_train_ens
#save_dict['TRAIN_Y'] = TRAIN_Y

np.save('{}RESULT_base2_lead{}_{}.npy'.format(filepath_vec, lead_name, model_tag), save_dict)
print('{}RESULT_base2_lead{}_{}.npy'.format(filepath_vec, lead_name, model_tag))



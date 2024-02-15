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


def verif_metric(VALID_target, Y_pred, ref):
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    metric = BS
    return metric / ref

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead1', help='lead1')
parser.add_argument('lead2', help='lead2')
parser.add_argument('lead3', help='lead3')
parser.add_argument('lead4', help='lead4')


args = vars(parser.parse_args())

# =============== #

lead1 = int(args['lead1'])
lead2 = int(args['lead2'])
lead3 = int(args['lead3'])
lead4 = int(args['lead4'])

lead_name = lead3 #args['lead_name']
model_tag = 'vgg2'
sigma = 2
L_vec = 4

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

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

lead = lead3
lead_window, flag_shift = neighbour_leads(lead)

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

label_smooth_v3 = record_v3

label_concat_v3 = np.sum(label_smooth_v3, axis=-1)
label_concat_v3[label_concat_v3>1] = 1

shape_label_v3 = label_concat_v3.shape
label_final_v3 = np.empty(shape_label_v3)

for i in range(shape_label_v3[0]):
    label_final_v3[i, ...] = scipy.ndimage.gaussian_filter(label_concat_v3[i, ...], sigma=sigma)

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
record_v4x[...] = 0.0 #np.nan

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

label_smooth_v4x = record_v4x

label_concat_v4x = np.sum(label_smooth_v4x, axis=-1)
label_concat_v4x[label_concat_v4x>1] = 1

shape_label_v4x = label_concat_v4x.shape
label_final_v4x = np.empty(shape_label_v4x)

for i in range(shape_label_v4x[0]):
    label_final_v4x[i, ...] = scipy.ndimage.gaussian_filter(label_concat_v4x[i, ...], sigma=sigma)
    
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

label_smooth_v4 = record_v4

label_concat_v4 = np.sum(label_smooth_v4, axis=-1)
label_concat_v4[label_concat_v4>1] = 1

shape_label_v4 = label_concat_v4.shape
label_final_v4 = np.empty(shape_label_v4)

for i in range(shape_label_v4[0]):
    label_final_v4[i, ...] = scipy.ndimage.gaussian_filter(label_concat_v4[i, ...], sigma=sigma)
    
filenames_train_v3 = np.load('/glade/work/ksha/NCAR/filenames_for_TRAIN_inds.npy', allow_pickle=True)[()]
filenames_valid_v3 = np.load('/glade/work/ksha/NCAR/filenames_for_VALID_inds.npy', allow_pickle=True)[()]

filenames_train_v4x = np.load('/glade/work/ksha/NCAR/filenames_for_TRAIN_inds_v4x.npy', allow_pickle=True)[()]
filenames_valid_v4x = np.load('/glade/work/ksha/NCAR/filenames_for_VALID_inds_v4x.npy', allow_pickle=True)[()]

filename_train_lead1_v3 = filenames_train_v3['lead{}'.format(lead1)]
filename_train_lead2_v3 = filenames_train_v3['lead{}'.format(lead2)]
filename_train_lead3_v3 = filenames_train_v3['lead{}'.format(lead3)]
filename_train_lead4_v3 = filenames_train_v3['lead{}'.format(lead4)]

filename_valid_lead1_v3 = filenames_valid_v3['lead{}'.format(lead1)]
filename_valid_lead2_v3 = filenames_valid_v3['lead{}'.format(lead2)]
filename_valid_lead3_v3 = filenames_valid_v3['lead{}'.format(lead3)]
filename_valid_lead4_v3 = filenames_valid_v3['lead{}'.format(lead4)]

filename_train_lead1_v4x = filenames_train_v4x['lead{}'.format(lead1)]
filename_train_lead2_v4x = filenames_train_v4x['lead{}'.format(lead2)]
filename_train_lead3_v4x = filenames_train_v4x['lead{}'.format(lead3)]
filename_train_lead4_v4x = filenames_train_v4x['lead{}'.format(lead4)]

filename_valid_lead1_v4x = filenames_valid_v4x['lead{}'.format(lead1)]
filename_valid_lead2_v4x = filenames_valid_v4x['lead{}'.format(lead2)]
filename_valid_lead3_v4x = filenames_valid_v4x['lead{}'.format(lead3)]
filename_valid_lead4_v4x = filenames_valid_v4x['lead{}'.format(lead4)]

# ============================================================ #
# Consistency check indices

IND_TRAIN_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead_{}{}{}{}.npy'.format(lead1, lead2, lead3, lead4), allow_pickle=True)[()]
TRAIN_ind1_v3 = IND_TRAIN_lead['lead{}'.format(lead1)]
TRAIN_ind2_v3 = IND_TRAIN_lead['lead{}'.format(lead2)]
TRAIN_ind3_v3 = IND_TRAIN_lead['lead{}'.format(lead3)]
TRAIN_ind4_v3 = IND_TRAIN_lead['lead{}'.format(lead4)]

IND_VALID_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead_{}{}{}{}.npy'.format(lead1, lead2, lead3, lead4), allow_pickle=True)[()]
VALID_ind1_v3 = IND_VALID_lead['lead{}'.format(lead1)]
VALID_ind2_v3 = IND_VALID_lead['lead{}'.format(lead2)]
VALID_ind3_v3 = IND_VALID_lead['lead{}'.format(lead3)]
VALID_ind4_v3 = IND_VALID_lead['lead{}'.format(lead4)]

IND_TRAIN_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead_{}{}{}{}_v4x.npy'.format(lead1, lead2, lead3, lead4), allow_pickle=True)[()]
TRAIN_ind1_v4x = IND_TRAIN_lead['lead{}'.format(lead1)]
TRAIN_ind2_v4x = IND_TRAIN_lead['lead{}'.format(lead2)]
TRAIN_ind3_v4x = IND_TRAIN_lead['lead{}'.format(lead3)]
TRAIN_ind4_v4x = IND_TRAIN_lead['lead{}'.format(lead4)]

IND_VALID_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead_{}{}{}{}_v4x.npy'.format(lead1, lead2, lead3, lead4), allow_pickle=True)[()]
VALID_ind1_v4x = IND_VALID_lead['lead{}'.format(lead1)]
VALID_ind2_v4x = IND_VALID_lead['lead{}'.format(lead2)]
VALID_ind3_v4x = IND_VALID_lead['lead{}'.format(lead3)]
VALID_ind4_v4x = IND_VALID_lead['lead{}'.format(lead4)]

data_lead1_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p3 = np.load('{}TRAIN_v3_vec_lead{}_part3_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]

data_lead2_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p3 = np.load('{}TRAIN_v3_vec_lead{}_part3_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

data_lead3_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p3 = np.load('{}TRAIN_v3_vec_lead{}_part3_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]

data_lead4_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p3 = np.load('{}TRAIN_v3_vec_lead{}_part3_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

TRAIN_lead1_v3 = np.concatenate((data_lead1_p0['y_vector'], data_lead1_p1['y_vector'], data_lead1_p2['y_vector'], data_lead1_p3['y_vector']), axis=0)
TRAIN_lead2_v3 = np.concatenate((data_lead2_p0['y_vector'], data_lead2_p1['y_vector'], data_lead2_p2['y_vector'], data_lead2_p3['y_vector']), axis=0)
TRAIN_lead3_v3 = np.concatenate((data_lead3_p0['y_vector'], data_lead3_p1['y_vector'], data_lead3_p2['y_vector'], data_lead3_p3['y_vector']), axis=0)
TRAIN_lead4_v3 = np.concatenate((data_lead4_p0['y_vector'], data_lead4_p1['y_vector'], data_lead4_p2['y_vector'], data_lead4_p3['y_vector']), axis=0)

TRAIN_lead1_y_v3 = np.concatenate((data_lead1_p0['y_true'], data_lead1_p1['y_true'], data_lead1_p2['y_true'], data_lead1_p3['y_true']), axis=0)
TRAIN_lead2_y_v3 = np.concatenate((data_lead2_p0['y_true'], data_lead2_p1['y_true'], data_lead2_p2['y_true'], data_lead2_p3['y_true']), axis=0)
TRAIN_lead3_y_v3 = np.concatenate((data_lead3_p0['y_true'], data_lead3_p1['y_true'], data_lead3_p2['y_true'], data_lead3_p3['y_true']), axis=0)
TRAIN_lead4_y_v3 = np.concatenate((data_lead4_p0['y_true'], data_lead4_p1['y_true'], data_lead4_p2['y_true'], data_lead4_p3['y_true']), axis=0)

# =========================================================== #
# Load feature vectors (HRRR v3, validation)

data_lead1_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead2_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead3_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead4_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

VALID_lead1_v3 = data_lead1_valid['y_vector']
VALID_lead2_v3 = data_lead2_valid['y_vector']
VALID_lead3_v3 = data_lead3_valid['y_vector']
VALID_lead4_v3 = data_lead4_valid['y_vector']

VALID_lead1_y_v3 = data_lead1_valid['y_true']
VALID_lead2_y_v3 = data_lead2_valid['y_true']
VALID_lead3_y_v3 = data_lead3_valid['y_true']
VALID_lead4_y_v3 = data_lead4_valid['y_true']

data_lead1_p0 = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead2_p0 = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead3_p0 = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead4_p0 = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

TRAIN_lead1_v4x = data_lead1_p0['y_vector']
TRAIN_lead2_v4x = data_lead2_p0['y_vector']
TRAIN_lead3_v4x = data_lead3_p0['y_vector']
TRAIN_lead4_v4x = data_lead4_p0['y_vector']

TRAIN_lead1_y_v4x = data_lead1_p0['y_true']
TRAIN_lead2_y_v4x = data_lead2_p0['y_true']
TRAIN_lead3_y_v4x = data_lead3_p0['y_true']
TRAIN_lead4_y_v4x = data_lead4_p0['y_true']

# =========================================================== #
# Load feature vectors (HRRR v4x, validation)

data_lead1_valid = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead2_valid = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead3_valid = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead4_valid = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

VALID_lead1_v4x = data_lead1_valid['y_vector']
VALID_lead2_v4x = data_lead2_valid['y_vector']
VALID_lead3_v4x = data_lead3_valid['y_vector']
VALID_lead4_v4x = data_lead4_valid['y_vector']

VALID_lead1_y_v4x = data_lead1_valid['y_true']
VALID_lead2_y_v4x = data_lead2_valid['y_true']
VALID_lead3_y_v4x = data_lead3_valid['y_true']
VALID_lead4_y_v4x = data_lead4_valid['y_true']

# ================================================================= #
# Collect feature vectors from all batch files (HRRR v3, validation)

L = len(TRAIN_ind2_v3)

filename_train1_pick_v3 = []
filename_train2_pick_v3 = []
filename_train3_pick_v3 = []
filename_train4_pick_v3 = []

TRAIN_v3_lead1 = np.empty((L, 128))
TRAIN_v3_lead2 = np.empty((L, 128))
TRAIN_v3_lead3 = np.empty((L, 128))
TRAIN_v3_lead4 = np.empty((L, 128))

TRAIN_Y_v3 = np.empty(L)

for i in range(L):
    
    ind_lead1_v3 = int(TRAIN_ind1_v3[i])
    ind_lead2_v3 = int(TRAIN_ind2_v3[i])
    ind_lead3_v3 = int(TRAIN_ind3_v3[i])
    ind_lead4_v3 = int(TRAIN_ind4_v3[i])
    
    filename_train1_pick_v3.append(filename_train_lead1_v3[ind_lead1_v3])
    filename_train2_pick_v3.append(filename_train_lead2_v3[ind_lead2_v3])
    filename_train3_pick_v3.append(filename_train_lead3_v3[ind_lead3_v3])
    filename_train4_pick_v3.append(filename_train_lead4_v3[ind_lead4_v3])
    
    TRAIN_v3_lead1[i, :] = TRAIN_lead1_v3[ind_lead1_v3, :]
    TRAIN_v3_lead2[i, :] = TRAIN_lead2_v3[ind_lead2_v3, :]
    TRAIN_v3_lead3[i, :] = TRAIN_lead3_v3[ind_lead3_v3, :]
    TRAIN_v3_lead4[i, :] = TRAIN_lead4_v3[ind_lead4_v3, :]
    
    TRAIN_Y_v3[i] = TRAIN_lead3_y_v3[ind_lead3_v3]
    
indx_train_v3, indy_train_v3, day_train_v3, lead_train_v3 = filename_to_loc(filename_train3_pick_v3)


# ================================================================= #
# Collect feature vectors from all batch files (HRRR v4x, validation)

L = len(TRAIN_ind2_v4x)

filename_train1_pick_v4x = []
filename_train2_pick_v4x = []
filename_train3_pick_v4x = []
filename_train4_pick_v4x = []

TRAIN_v4x_lead1 = np.empty((L, 128))
TRAIN_v4x_lead2 = np.empty((L, 128))
TRAIN_v4x_lead3 = np.empty((L, 128))
TRAIN_v4x_lead4 = np.empty((L, 128))

TRAIN_Y_v4x = np.empty(L)

for i in range(L):
    
    ind_lead1_v4x = int(TRAIN_ind1_v4x[i])
    ind_lead2_v4x = int(TRAIN_ind2_v4x[i])
    ind_lead3_v4x = int(TRAIN_ind3_v4x[i])
    ind_lead4_v4x = int(TRAIN_ind4_v4x[i])
    
    filename_train1_pick_v4x.append(filename_train_lead1_v4x[ind_lead1_v4x])
    filename_train2_pick_v4x.append(filename_train_lead2_v4x[ind_lead2_v4x])
    filename_train3_pick_v4x.append(filename_train_lead3_v4x[ind_lead3_v4x])
    filename_train4_pick_v4x.append(filename_train_lead4_v4x[ind_lead4_v4x])
    
    TRAIN_v4x_lead1[i, :] = TRAIN_lead1_v4x[ind_lead1_v4x, :]
    TRAIN_v4x_lead2[i, :] = TRAIN_lead2_v4x[ind_lead2_v4x, :]
    TRAIN_v4x_lead3[i, :] = TRAIN_lead3_v4x[ind_lead3_v4x, :]
    TRAIN_v4x_lead4[i, :] = TRAIN_lead4_v4x[ind_lead4_v4x, :]
    
    TRAIN_Y_v4x[i] = TRAIN_lead3_y_v4x[ind_lead3_v4x]
    
indx_train_v4x, indy_train_v4x, day_train_v4x, lead_train_v4x = filename_to_loc(filename_train3_pick_v4x)


# ================================================================== #
# Collect feature vectors from all batch files (HRRR v3, validation)
L = len(VALID_ind2_v3)

filename_valid1_pick_v3 = []
filename_valid2_pick_v3 = []
filename_valid3_pick_v3 = []
filename_valid4_pick_v3 = []

VALID_v3_lead1 = np.empty((L, 128))
VALID_v3_lead2 = np.empty((L, 128))
VALID_v3_lead3 = np.empty((L, 128))
VALID_v3_lead4 = np.empty((L, 128))

VALID_Y_v3 = np.empty(L)

for i in range(L):
    
    ind_lead1_v3 = int(VALID_ind1_v3[i])
    ind_lead2_v3 = int(VALID_ind2_v3[i])
    ind_lead3_v3 = int(VALID_ind3_v3[i])
    ind_lead4_v3 = int(VALID_ind4_v3[i])
    
    filename_valid1_pick_v3.append(filename_valid_lead1_v3[ind_lead1_v3])
    filename_valid2_pick_v3.append(filename_valid_lead2_v3[ind_lead2_v3])
    filename_valid3_pick_v3.append(filename_valid_lead3_v3[ind_lead3_v3])
    filename_valid4_pick_v3.append(filename_valid_lead4_v3[ind_lead4_v3])
    
    VALID_v3_lead1[i, :] = VALID_lead1_v3[ind_lead1_v3, :]
    VALID_v3_lead2[i, :] = VALID_lead2_v3[ind_lead2_v3, :]
    VALID_v3_lead3[i, :] = VALID_lead3_v3[ind_lead3_v3, :]
    VALID_v3_lead4[i, :] = VALID_lead4_v3[ind_lead4_v3, :]
    
    VALID_Y_v3[i] = VALID_lead3_y_v3[ind_lead3_v3]
    
indx_valid_v3, indy_valid_v3, day_valid_v3, lead_valid_v3 = filename_to_loc(filename_valid3_pick_v3)


L = len(VALID_ind2_v4x)

filename_valid1_pick_v4x = []
filename_valid2_pick_v4x = []
filename_valid3_pick_v4x = []
filename_valid4_pick_v4x = []

VALID_v4x_lead1 = np.empty((L, 128))
VALID_v4x_lead2 = np.empty((L, 128))
VALID_v4x_lead3 = np.empty((L, 128))
VALID_v4x_lead4 = np.empty((L, 128))

VALID_Y_v4x = np.empty(L)

for i in range(L):
    
    ind_lead1_v4x = int(VALID_ind1_v4x[i])
    ind_lead2_v4x = int(VALID_ind2_v4x[i])
    ind_lead3_v4x = int(VALID_ind3_v4x[i])
    ind_lead4_v4x = int(VALID_ind4_v4x[i])
    
    filename_valid1_pick_v4x.append(filename_valid_lead1_v4x[ind_lead1_v4x])
    filename_valid2_pick_v4x.append(filename_valid_lead2_v4x[ind_lead2_v4x])
    filename_valid3_pick_v4x.append(filename_valid_lead3_v4x[ind_lead3_v4x])
    filename_valid4_pick_v4x.append(filename_valid_lead4_v4x[ind_lead4_v4x])
    
    VALID_v4x_lead1[i, :] = VALID_lead1_v4x[ind_lead1_v4x, :]
    VALID_v4x_lead2[i, :] = VALID_lead2_v4x[ind_lead2_v4x, :]
    VALID_v4x_lead3[i, :] = VALID_lead3_v4x[ind_lead3_v4x, :]
    VALID_v4x_lead4[i, :] = VALID_lead4_v4x[ind_lead4_v4x, :]
    
    VALID_Y_v4x[i] = VALID_lead3_y_v4x[ind_lead3_v4x]
    
ALL_TRAIN_v3 = np.concatenate((TRAIN_v3_lead1[:, None, :], TRAIN_v3_lead2[:, None, :], TRAIN_v3_lead3[:, None, :],), axis=1)
ALL_VALID_v3 = np.concatenate((VALID_v3_lead1[:, None, :], VALID_v3_lead2[:, None, :], VALID_v3_lead3[:, None, :],), axis=1)

ALL_TRAIN_v4x = np.concatenate((TRAIN_v4x_lead1[:, None, :], TRAIN_v4x_lead2[:, None, :], TRAIN_v4x_lead3[:, None, :],), axis=1)
ALL_VALID_v4x = np.concatenate((VALID_v4x_lead1[:, None, :], VALID_v4x_lead2[:, None, :], VALID_v4x_lead3[:, None, :],), axis=1)

TRAIN_VEC = np.concatenate((ALL_TRAIN_v3, ALL_TRAIN_v4x, ), axis=0)
VALID_VEC = np.concatenate((ALL_VALID_v3, ALL_VALID_v4x), axis=0)

TRAIN_Y = np.concatenate((TRAIN_Y_v3, TRAIN_Y_v4x), axis=0)
VALID_Y = np.concatenate((VALID_Y_v3, VALID_Y_v4x), axis=0)

TRAIN_pos_x = TRAIN_VEC[TRAIN_Y==1]
TRAIN_neg_x = TRAIN_VEC[TRAIN_Y==0]

lon_norm, lat_norm, elev_norm = feature_extract(filename_train2_pick_v3+filename_train2_pick_v4x, 
                                                lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)
TRAIN_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None]), axis=1)

lon_norm, lat_norm, elev_norm = feature_extract(filename_valid2_pick_v3+filename_valid2_pick_v4x, 
                                                lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)
VALID_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None]), axis=1)

TRAIN_stn_pos = TRAIN_stn[TRAIN_Y==1]
TRAIN_stn_neg = TRAIN_stn[TRAIN_Y==0]

TRAIN_Y_pos = TRAIN_Y[TRAIN_Y==1]
TRAIN_Y_neg = TRAIN_Y[TRAIN_Y==0]

TRAIN_stn_pos_ = TRAIN_stn_pos
TRAIN_stn_neg_ = TRAIN_stn_neg

TRAIN_pos_x_ = TRAIN_pos_x
TRAIN_neg_x_ = TRAIN_neg_x

TRAIN_Y_pos_ = TRAIN_Y_pos
TRAIN_Y_neg_ = TRAIN_Y_neg

seeds = [12342, 2536234, 98765, 473, 865, 7456, 69472, 3456357, 3425, 678,
         2452624, 5787, 235362, 67896, 98454, 12445, 46767, 78906, 345, 8695, 
         2463725, 4734, 23234, 884, 2341, 362, 5, 234, 483, 785356, 23425, 3621, 
         58461, 80968765, 123, 425633, 5646, 67635, 76785, 34214, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


# ============================================== #
ref = np.sum(VALID_Y) / len(VALID_Y)
grid_shape = lon_80km.shape

# ========== Training loop ========== #
L_pos = len(TRAIN_stn_pos_)
L_neg = len(TRAIN_stn_neg_)
# =========== Model Section ========== #

tol = 0
min_del = 0
max_tol = 10
epochs = 500
batch_size = 64
L_train = 16

def create_classif_head(L_vec, k_size=2, padding='same'):
    
    IN = keras.Input((L_vec, 128))
    X = IN
    X = keras.layers.Conv1D(128, kernel_size=k_size, strides=1, padding=padding)(X)
    X = keras.layers.Activation("gelu")(X)
    #
    IN_vec = keras.Input((2,))
    
    X = keras.layers.GlobalMaxPool1D()(X) #X = keras.layers.Flatten()(X)
    X = keras.layers.Concatenate()([X, IN_vec])

    X = keras.layers.Dropout(0.025)(X, training=True)
    
    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=[IN, IN_vec], outputs=OUT)
    return model

for i, seed in enumerate(seeds):
    
    mu.set_seeds(seed)
    
    record = 1.1
    
    key = '{}_lead{}_mc{}'.format(model_tag, lead_name, i)
    model_name = '{}'.format(key)
    model_path = temp_dir+model_name
        
    # ================================== #
    model = create_classif_head(L_vec)
    # ================================== #
    
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=keras.optimizers.Adam(lr=1e-4))

    for i in range(epochs):            
        start_time = time.time()

        # loop of batch
        for j in range(L_train):
            N_pos = 32
            N_neg = batch_size - N_pos

            ind_neg = du.shuffle_ind(L_neg)
            ind_pos = du.shuffle_ind(L_pos)

            ind_neg_pick = ind_neg[:N_neg]
            ind_pos_pick = ind_pos[:N_pos]

            X_batch_neg = TRAIN_neg_x_[ind_neg_pick, :]
            X_batch_pos = TRAIN_pos_x_[ind_pos_pick, :]
            
            X_batch_stn_neg = TRAIN_stn_neg_[ind_neg_pick, :]
            X_batch_stn_pos = TRAIN_stn_pos_[ind_pos_pick, :]
            
            Y_batch_neg = TRAIN_Y_neg_[ind_neg_pick]
            Y_batch_pos = TRAIN_Y_pos_[ind_pos_pick]
            
            X_batch = np.concatenate((X_batch_neg, X_batch_pos), axis=0)
            X_batch_stn = np.concatenate((X_batch_stn_neg, X_batch_stn_pos), axis=0)
            Y_batch = np.concatenate((Y_batch_neg, Y_batch_pos), axis=0)

            ind_ = du.shuffle_ind(batch_size)

            X_batch = X_batch[ind_, :]
            X_batch_stn = X_batch_stn[ind_, :]
            Y_batch = Y_batch[ind_]

            # train on batch
            model.train_on_batch([X_batch, X_batch_stn], Y_batch);

        # epoch end operations
        Y_pred = model.predict([VALID_VEC, VALID_stn])

        Y_pred[Y_pred<0] = 0
        Y_pred[Y_pred>1] = 1

        record_temp = verif_metric(VALID_Y, Y_pred, ref)

        # if i % 10 == 0:
        #     model.save(model_path_backup)

        if (record - record_temp > min_del):
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            tol = 0
            
            #print('tol: {}'.format(tol))
            # save
            print('save to: {}'.format(model_path))
            model.save(model_path)
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
            if record_temp > 1.0:
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
        

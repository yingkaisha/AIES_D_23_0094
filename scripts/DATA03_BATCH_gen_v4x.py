import os
import sys
from glob import glob

import time
import h5py
import zarr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

from datetime import datetime, timedelta

import dask.array as da

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============== #
lead = int(args['lead'])
print('Batch generation script for lead {}'.format(lead))

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

# ind_pick = [0, 1, 3, 4, 8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22] 

# log_norm = [True, False, True, True, True, True, True, True, True, False, False, 
#             False, False, True, True, True, True, False, False, False, False, False, False]

# sparse = [True, False, True, True, True, True, True, True, True, False, False, 
#           False, False, False, True, True, True, False, False, False, False, False, False]

ind_pick = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
log_norm = [True, False, True, True, True, False, False, 
            True, True, True, True, False, False, False, False]

print('Loading HRRR ...')
HRRRv4x_lead = zarr.load(save_dir_campaign+'HRRR_{:02}_v4x.zarr'.format(lead))
print('... done')

print('Loading observation on lead {} ...'.format(lead))

lead_window, flag_shift = neighbour_leads(lead)

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

print('... done')
                        
print('Loading Geographical information ...')

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]
    
print('... done')

shape_80km = lon_80km.shape
shape_3km = lon_3km.shape
shape_record = record_v4x.shape

print('Map 80-km grids to 3-km HRRR domain ...')

indx_array = np.empty(shape_80km)
indy_array = np.empty(shape_80km)

gridTree = cKDTree(list(zip(lon_3km.ravel(), lat_3km.ravel()))) #KDTree_wraper(xgrid, ygrid)

for xi in range(shape_80km[0]):
    for yi in range(shape_80km[1]):
        
        temp_lon = lon_80km[xi, yi]
        temp_lat = lat_80km[xi, yi]
        
        dist, indexes = gridTree.query(list(zip(np.array(temp_lon)[None], np.array(temp_lat)[None])))
        indx_3km, indy_3km = np.unravel_index(indexes, shape_3km)
        
        indx_array[xi, yi] = indx_3km[0]
        indy_array[xi, yi] = indy_3km[0]
      
print('... done')
    
base_ref = datetime(2019, 10, 1)
date_list_v4x = [base_ref + timedelta(days=day) for day in range(429)]

L_train = shape_record[0]

input_size = 64
half_margin = 32

L_vars = len(ind_pick)
L_vars_per = len(ind_pick)

out_slice = np.empty((1, input_size, input_size, L_vars))

#batch_dir = '/glade/campaign/cisl/aiml/ksha/NCAR_batch/'
batch_dir = path_batch_v4x #'/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
prefix = '{}_day{:03d}_{}_{}_{}_indx{}_indy{}_lead{}.npy'

flag_torn = 'neg'
flag_wind = 'neg'
flag_hail = 'neg'

if lead == 2:
    leads = [2, 3, 4]
else:
    leads = [lead-1, lead, lead+1]

norm_stats = np.load('/glade/work/ksha/NCAR/stats_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]))
max_stats = np.load('/glade/work/ksha/NCAR/p90_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]))

print('Generating batches ...')

#L_train
for day in range(L_train):
    if day > 156:
        tv_label = 'VALID'
    else:
        tv_label = 'TRAIN'
        
    for ix in range(shape_80km[0]):
        for iy in range(shape_80km[1]):
            
            indx = int(indx_array[ix, iy])
            indy = int(indy_array[ix, iy])
            
            x_edge_left_ = indx - half_margin
            x_edge_right_ = indx + half_margin
            y_edge_bottom_ = indy - half_margin
            y_edge_top_ = indy + half_margin

            # # indices must be valid
            x_edge_left = x_edge_left_
            y_edge_bottom = y_edge_bottom_
            x_edge_right = x_edge_right_
            y_edge_top = y_edge_top_

            if land_mask_80km[ix, iy]:
                
                obs_temp = record_v4x[day, ix, iy, :]

                if obs_temp[0] == 0:
                    flag_torn = 'neg'
                else:
                    flag_torn = 'pos'

                if obs_temp[1] == 0:
                    flag_wind = 'neg'
                else:
                    flag_wind = 'pos'

                if obs_temp[2] == 0:
                    flag_hail = 'neg'
                else:
                    flag_hail = 'pos' 

                save_name = batch_dir+prefix.format(tv_label, day, flag_torn, flag_wind, flag_hail, ix, iy, lead)

                if os.path.isfile(save_name):
                    #print('{} Exists'.format(save_name))
                    continue;
                else:
                    # current day HRRR data
                    HRRRv4x_lead_ = HRRRv4x_lead[day, ...]

                    if x_edge_left_ < 0:
                        delta_x = x_edge_left_
                        x_edge_left = 0
                        x_edge_right = x_edge_right_ - delta_x

                    if y_edge_bottom_ < 0:
                        delta_y = y_edge_bottom_
                        y_edge_bottom = 0
                        y_edge_top = y_edge_top_ - delta_y

                    if x_edge_right > shape_3km[0]:
                        HRRRv4x_lead_ = np.pad(HRRRv4x_lead_, ((0, 128), (0, 0), (0, 0)), mode='constant', constant_values=0)

                    if y_edge_top > shape_3km[1]:
                        HRRRv4x_lead_ = np.pad(HRRRv4x_lead_, ((0, 0), (0, 128), (0, 0)), mode='constant', constant_values=0)

                    hrrr_3km = HRRRv4x_lead_[x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]

                    means = norm_stats[ix, iy, :, 0]
                    stds = norm_stats[ix, iy, :, 1]
                    max_vals = max_stats[ix, iy, :, 3]

                    for v, ind_var in enumerate(ind_pick):

                        temp = hrrr_3km[..., ind_var]

                        if ind_var == 0:
                            temp[temp<0] = 0

                        if ind_var == 16:
                            temp = -1*temp
                            temp[temp<0] = 0

                        if log_norm[ind_var]:
                            temp = np.log(np.abs(temp)+1)

                            if v < 9:
                                temp = temp/stds[v]/max_vals[v]
                            else:
                                temp = 3.0*temp/stds[v]/max_vals[v]

                        else:
                            temp = (temp - means[v])/stds[v]

                        out_slice[..., v] = temp

                    if np.sum(np.isnan(out_slice)) > 0:
                        #print('nanananannana')
                        continue;
                    else:
                        print(save_name)
                        np.save(save_name, out_slice)

print('... done')
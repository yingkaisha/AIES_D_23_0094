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

#import dask.array as da

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]

shape_3km = land_mask_3km.shape
shape_80km = land_mask_80km.shape

ind_pick = [0, 1, 3, 4, 8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22]
log_norm = [True, False, True, True, True, True, True, True, True, False, False, 
            False, False, True, True, True, True, False, False, False, False, False, False]

# ===== Combine mean and std ===== #

file_path = '/glade/work/ksha/NCAR/stats_v4_80km_ix{}_iy{}_lead{}{}{}.npy'

# LEADs = [[2, 3, 4], [3, 4, 5], [4, 5, 6], 
#          [5, 6, 7], [6, 7, 8], [7, 8, 9], 
#          [8, 9, 10], [9, 10, 11], [10, 11, 12], 
#          [11, 12, 13], [12, 13, 14], [13, 14, 15], 
#          [14, 15, 16], [15, 16, 17], [16, 17, 18], 
#          [17, 18, 19], [18, 19, 20], [19, 20, 21], 
#          [20, 21, 22], [21, 22, 23], [22, 23, 24]]

LEADs = [[23, 24, 25], [24, 25, 26], [25, 26, 27], 
         [26, 27, 28], [27, 28, 29], [28, 29, 30],
         [29, 30, 31], [30, 31, 32], [31, 32, 33],
         [32, 33, 34], [33, 34, 35], [34, 35, 36]]

for i in range(len(LEADs)):
    leads = LEADs[i]

    out_lead = np.empty(shape_80km+(15, 2))
    out_lead[...] = np.nan

    for ix in range(shape_80km[0]):
        for iy in range(shape_80km[1]):
            if land_mask_80km[ix, iy]:
                temp = np.load(file_path.format(ix, iy, leads[0], leads[1], leads[2]))
                out_lead[ix, iy, ...] = temp
            else:
                out_lead[ix, iy, ...] = np.nan
                
    print('/glade/work/ksha/NCAR/stats_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]))
    np.save('/glade/work/ksha/NCAR/stats_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]), out_lead)
    

file_path = '/glade/work/ksha/NCAR/p90_v4_80km_ix{}_iy{}_lead{}{}{}.npy'
    
for i in range(len(LEADs)):
    leads = LEADs[i]

    out_lead = np.empty(shape_80km+(15, 4))
    out_lead[...] = np.nan

    for ix in range(shape_80km[0]):
        for iy in range(shape_80km[1]):
            if land_mask_80km[ix, iy]:
                temp = np.load(file_path.format(ix, iy, leads[0], leads[1], leads[2]))
                out_lead[ix, iy, ...] = temp
            else:
                out_lead[ix, iy, ...] = np.nan
                
    print('/glade/work/ksha/NCAR/p90_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]))
    np.save('/glade/work/ksha/NCAR/p90_allv4_80km_full_lead{}{}{}.npy'.format(leads[0], leads[1], leads[2]), out_lead)
    



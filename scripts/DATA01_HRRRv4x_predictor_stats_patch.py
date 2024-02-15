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
parser.add_argument('lead0', help='lead')
parser.add_argument('lead1', help='lead')
parser.add_argument('lead2', help='lead')
args = vars(parser.parse_args())

leads = [int(args['lead0']), int(args['lead1']), int(args['lead2'])]
#leads = [2, 3, 4]

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]

shape_3km = land_mask_3km.shape
shape_80km = land_mask_80km.shape
half_margin = 32

# ind_pick = [0, 1, 3, 4, 8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22]
# log_norm = [True, False, True, True, True, True, True, True, True, False, False, 
#             False, False, True, True, True, True, False, False, False, False, False, False]

ind_pick = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
log_norm = [True, False, True, True, True, False, False, 
            True, True, True, True, False, False, False, False]

# Compute mean and std from one year of forecasts (366 days)
HRRRv4x_lead_p0 = da.from_zarr(save_dir_campaign+'HRRR_{:02}_v4x.zarr'.format(leads[0]))[:366, ...]
HRRRv4x_lead_p1 = da.from_zarr(save_dir_campaign+'HRRR_{:02}_v4x.zarr'.format(leads[1]))[:366, ...]
HRRRv4x_lead_p2 = da.from_zarr(save_dir_campaign+'HRRR_{:02}_v4x.zarr'.format(leads[2]))[:366, ...]

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
        
for ix in range(shape_80km[0]):
    for iy in range(shape_80km[1]):

        indx = int(indx_array[ix, iy])
        indy = int(indy_array[ix, iy])
        
        x_edge_left_ = indx - half_margin
        x_edge_right_ = indx + half_margin

        y_edge_bottom_ = indy - half_margin
        y_edge_top_ = indy + half_margin
        
        # # indices must be valid
        x_edge_left = np.max([x_edge_left_, 0])
        y_edge_bottom = np.max([y_edge_bottom_, 0])
        x_edge_right = np.min([x_edge_right_, shape_3km[0]])
        y_edge_top = np.min([y_edge_top_, shape_3km[1]])

        if land_mask_80km[ix, iy]:
            save_name = '/glade/work/ksha/NCAR/stats_v4x_80km_ix{}_iy{}_lead{}{}{}.npy'.format(ix, iy, leads[0], leads[1], leads[2])
            # ------ only for re-run ----- #
            if os.path.isfile(save_name):
                print('{} Exists'.format(save_name))
            else:
                p0 = HRRRv4x_lead_p0[:, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                p1 = HRRRv4x_lead_p1[:, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                p2 = HRRRv4x_lead_p2[:, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]

                stats_var = np.empty((len(ind_pick), 2))
                stats_var[...] = np.nan

                for v, ind_var in enumerate(ind_pick):

                    p_all = np.concatenate((p0[..., ind_var].ravel(), p1[..., ind_var].ravel(), p2[..., ind_var].ravel()))
                    p_all = p_all[~np.isnan(p_all)]

                    if ind_var == 0:
                        p_all[p_all<0] = 0

                    # v4x, ind=10 is Convective Inhibition 
                    if ind_var == 10:
                        p_all = -1*p_all
                        p_all[p_all<0] = 0

                    p_all = p_all[~np.isnan(p_all)]

                    if log_norm[ind_var]:

                        if np.sum(np.isnan(p_all))>0:
                            waerg
                        if np.min(p_all)<0:
                            wqerga

                        temp = np.log(p_all+1)
                        stats_var[v, 0] = np.mean(temp)
                        stats_var[v, 1] = np.std(temp)
                    else:
                        stats_var[v, 0] = np.mean(p_all)
                        stats_var[v, 1] = np.std(p_all)

                # print("--- %s seconds ---" % (time.time() - start_time))
                np.save(save_name, stats_var)
                print(save_name)
                    
import sys
from glob import glob

import zarr
import pygrib
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============== #
lead = int(args['lead'])
print('subsetting {} hr HRRRv4x fcsts'.format(lead))

# Dr Sobash's storage
filepath = '/glade/campaign/mmm/parc/sobash/HRRRv4/{}/{}'

# grib var indices
HRRR_inds_p1 = [601, 614, 617, 621, 630, 671, 674, 679, 690, 708, 709, 736, 737, 742, 743]
HRRR_inds_p2 = [601, 614, 617, 621, 630, 671, 674, 679, 690, 708+4, 709+4, 736+4, 737+4, 742+4, 743+4]

base_ref = datetime(2019, 10, 1)
date_list_v4x = [base_ref + timedelta(days=day) for day in range(429)]

# obtain grib var name reference
i = 1
var_names_p1 = []
date_temp = date_list_v4x[i]

name_folder = datetime.strftime(date_temp, '%Y%m%d00')
filename_pattern = '*0000{:02d}00'.format(lead)
filename = sorted(glob(filepath.format(name_folder, filename_pattern)))[0]

with pygrib.open(filename) as grbio:
    var_list = grbio()
    for i, ind in enumerate(HRRR_inds_p1):
        var_names_p1.append(str(grbio[ind])[:35])
        
i = 241
var_names_p2 = []
date_temp = date_list_v4x[i]

name_folder = datetime.strftime(date_temp, '%Y%m%d00')
filename_pattern = '*0000{:02d}00'.format(lead)
filename = sorted(glob(filepath.format(name_folder, filename_pattern)))[0]

with pygrib.open(filename) as grbio:
    var_list = grbio()
    for i, ind in enumerate(HRRR_inds_p2):
        var_names_p2.append(str(grbio[ind])[:35])
        
# ----- #
prefix = 'v4x'
# ----- #

# Max number of possible files
L = len(date_list_v4x)

# Allocation (one array for one lead time)
VARs = np.empty((L, 1059, 1799, len(HRRR_inds_p1)))


for i in range(L):

    var_names_temp = []
    date_temp = date_list_v4x[i]
    print(date_temp)
    
    name_folder = datetime.strftime(date_temp, '%Y%m%d00')
    filename_pattern = '*0000{:02d}00'.format(lead)
    filename = sorted(glob(filepath.format(name_folder, filename_pattern)))

    if len(filename) > 0:
        filename = filename[0]
        # later half
        if i >= 241:
            with pygrib.open(filename) as grbio:
                var_list = grbio()
                for j, ind in enumerate(HRRR_inds_p2):
                    VARs[i, ..., j] = grbio[ind].values
                    var_names_temp.append(str(grbio[ind])[:35])
                    
            if var_names_temp != var_names_p2:
                print('var names mismatch')
                VARs[i, ...] = np.nan
        # first half       
        else:
            with pygrib.open(filename) as grbio:
                var_list = grbio()
                for j, ind in enumerate(HRRR_inds_p1):
                    VARs[i, ..., j] = grbio[ind].values
                    var_names_temp.append(str(grbio[ind])[:35])
                    
            if var_names_temp != var_names_p1:
                print('var names mismatch')
                VARs[i, ...] = np.nan
    else:
        print(filepath.format(name_folder, filename_pattern)+' is missing')
        VARs[i, ...] = np.nan

save_name = save_dir_campaign+'HRRR_{}_{}.zarr'.format(lead, prefix)
print('Save to {}'.format(save_name))
zarr.save(save_name, VARs)
        
        
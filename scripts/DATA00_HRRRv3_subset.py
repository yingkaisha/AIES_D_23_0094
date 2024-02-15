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
lead = args['lead']
print('subsetting {} hr HRRRv3 fcsts'.format(lead))

HRRR_dir = '/glade/campaign/cisl/aiml/ksha/HRRR/'

# Variables to extract

HRRRv3_inds = [1003, 1014, 1016, 1018, 1020, 1022, 1024, 1025, 1028, 1041, 1044, 
               1046, 1047, 1048, 1059, 1073, 1074, 1093, 1094, 1097, 1098, 1099, 1100]

HRRRv4_inds = [1003, 1014, 1016, 1018, 1020, 1022, 1024, 1025, 1028, 1041, 1044, 
               1047, 1048, 1049, 1060, 1074, 1075, 1097, 1098, 1101, 1102, 1103, 1104]

base_v3_s = datetime(2018, 7, 15)
base_v3_e = datetime(2020, 12, 2)

base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 15)

base_ref = datetime(2010, 1, 1)

date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180)]

i = 0
var_names_v3 = []
date_temp = date_list_v3[i]
with pygrib.open((datetime.strftime(date_temp, HRRR_dir+'fcst{}hr/HRRR.%Y%m%d.natf{}.grib2')).format(lead, lead)) as grbio:
    for i, ind in enumerate(HRRRv3_inds):
        var_names_v3.append(str(grbio[ind])[:35])

var_names_v4 = []
date_temp = date_list_v4[i]
with pygrib.open((datetime.strftime(date_temp, HRRR_dir+'fcst{}hr/HRRR.%Y%m%d.natf{}.grib2')).format(lead, lead)) as grbio:
    for i, ind in enumerate(HRRRv4_inds):
        var_names_v4.append(str(grbio[ind])[:35])

# ----- #
var_inds = HRRRv3_inds
var_names = var_names_v3
prefix = 'v3'
date_list = date_list_v3
# ----- #

# Max number of possible files
L = len(date_list)

# Allocation (one array for one lead time)
VARs = np.empty((L, 1059, 1799, 23))

for i in range(L):

    var_names_temp = []
    date_temp = date_list[i]

    try:
        with pygrib.open((datetime.strftime(date_temp, HRRR_dir+'fcst{}hr/HRRR.%Y%m%d.natf{}.grib2')).format(lead, lead)) as grbio:
            for j, ind in enumerate(var_inds):
                VARs[i, ..., j] = grbio[ind].values
                var_names_temp.append(str(grbio[ind])[:35])
                
        print('Success: {}'.format(date_temp))
        
    except:
        print('Missing file: {}'.format(date_temp))
        VARs[i, ...] = np.nan
        
    if var_names_temp == var_names is False:
        print('grib2 var table mismatch: {}'.format(date_temp))
        VARs[i, ...] = np.nan

save_name = save_dir_campaign+'HRRR_{}_{}.zarr'.format(lead, prefix)
print('Save to {}'.format(save_name))
zarr.save(save_name, VARs)
        
        
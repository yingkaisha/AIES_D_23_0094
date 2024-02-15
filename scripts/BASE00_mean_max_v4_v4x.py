'''
Compute the max and mean values from each 64-by-64 training batch and all its predictors.
The max and mean values are applied as predictors of an MLP baseline.

v4x mean max values are used for standardization.
v4 mean max values are used for comparisons.
'''

# general tools
import os
import re
import sys
import time
import numpy as np
from glob import glob

from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

lead = int(args['lead'])

def mean_max_extractor(filenames):
    
    L = len(filenames)
    OUT = np.empty((L, 30))

    for i, file in enumerate(filenames):
        data = np.load(file, allow_pickle=True)[0, ...]

        data_mean = np.mean(data, axis=(0, 1))
        data_max = np.max(data, axis=(0, 1))
        
        OUT[i, :15] = data_mean
        OUT[i, 15:] = data_max
        
    return OUT


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

filenames_train_v4x = np.load('/glade/work/ksha/NCAR/filenames_for_TRAIN_inds_v4x.npy', allow_pickle=True)[()]
filenames_valid_v4x = np.load('/glade/work/ksha/NCAR/filenames_for_VALID_inds_v4x.npy', allow_pickle=True)[()]

filename_train_lead_v4x = filenames_train_v4x['lead{}'.format(lead)]
filename_valid_lead_v4x = filenames_valid_v4x['lead{}'.format(lead)]
filename_test_lead = sorted(glob("{}TEST*lead{}.npy".format(path_name1_v4_test, lead)))

start_time = time.time()
mean_max = mean_max_extractor(filename_train_lead_v4x)
save_name = '/glade/work/ksha/NCAR/TRAIN_v4x_meanmax_lead{}.npy'.format(lead)
print(save_name)
np.save(save_name, mean_max)
print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
mean_max = mean_max_extractor(filename_valid_lead_v4x)
save_name = '/glade/work/ksha/NCAR/VALID_v4x_meanmax_lead{}.npy'.format(lead)
print(save_name)
np.save(save_name, mean_max)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
mean_max = mean_max_extractor(filename_test_lead)
save_name = '/glade/work/ksha/NCAR/TEST_v4_meanmax_lead{}.npy'.format(lead)
print(save_name)
np.save(save_name, mean_max)
print("--- %s seconds ---" % (time.time() - start_time))

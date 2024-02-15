'''
Convert testing set 64-by-64 input frames into feature vectors.
'''

# general tools
import sys
from glob import glob

# data tools
import time
import h5py
import numpy as np

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
parser.add_argument('model', help='model')
parser.add_argument('prefix', help='prefix')
args = vars(parser.parse_args())

lead = int(args['lead'])
model_name = args['model']
prefix = args['prefix']
N_vars = L_vars = 15

# Collection batch names from scratch and campaign
names = glob("{}TEST*lead{}.npy".format(path_batch_v4, lead))
filename_valid = sorted(names)
print(len(filename_valid))

#filename_valid = filename_valid[:100]

# Combine individual batch files to a large array
L_valid = len(filename_valid)
L_var = L_vars

TEST_input_64 = np.empty((L_valid, 64, 64, L_var))
TEST_target = np.ones(L_valid)

for i, name in enumerate(filename_valid):
    data = np.load(name, allow_pickle=True)
    for k, c in enumerate(ind_pick_from_batch):
        
        TEST_input_64[i, ..., k] = data[..., c]

        if 'pos' in name:
            TEST_target[i] = 1.0
        else:
            TEST_target[i] = 0.0

# Crerate model
# model = mu.create_model_base(input_shape=(64, 64, 15), depths=[3, 3, 27, 3], projection_dims=[32, 64, 96, 128], first_pool=4)
model = mu.create_model_vgg(input_shape=(64, 64, 15), channels=[48, 64, 96, 128])

# get current weights
W_new = model.get_weights()

# get stored weights
print('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_name))
W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_name))

# update stored weights to new weights
for i in range(len(W_new)):
    if W_new[i].shape == W_old[i].shape:
        W_new[i] = W_old[i]
    else:
        # the size of the weights always match
        ewraewthws

model.set_weights(W_new)

# predict feature vectors
Y_vector = model.predict([TEST_input_64,])

save_dict = {}
save_dict['y_true'] = TEST_target
save_dict['y_vector'] = Y_vector
save_name = "/glade/work/ksha/NCAR/TEST_v4_vec_lead{}_{}.npy".format(lead, prefix)
print(save_name)
np.save(save_name, save_dict)


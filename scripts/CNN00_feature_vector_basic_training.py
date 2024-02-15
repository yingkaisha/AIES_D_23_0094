'''
Convert training set 64-by-64 input frames into feature vectors.
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
parser.add_argument('part', help='part')
parser.add_argument('lead', help='lead')
parser.add_argument('model', help='model')
parser.add_argument('prefix', help='prefix')
parser.add_argument('ver', help='ver')
args = vars(parser.parse_args())

# =============== #
part = int(args['part'])
lead = int(args['lead'])
model_name = args['model']
prefix = args['prefix']
ver = args['ver']
gap = 200000
N_vars = L_vars = 15
# =============== #

# Collection batch names from scratch and campaign
names = glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_{}_temp/TRAIN*lead{}.npy".format(ver, lead))

filename_train = sorted(names)

# divide batch files into three parts
if part == 0:
    filename_train = filename_train[:gap]
elif part == 1:
    filename_train = filename_train[gap:2*gap]
elif part == 2:
    filename_train = filename_train[2*gap:3*gap]
else:
    filename_train = filename_train[3*gap:]

L_train = len(filename_train)
L_var = L_vars

# Combine individual batch files to a large array
TEST_input_64 = np.empty((L_train, 64, 64, L_var))
TEST_target = np.ones(L_train)

for i, name in enumerate(filename_train):
    data = np.load(name, allow_pickle=True)
    for k, c in enumerate(ind_pick_from_batch):
        
        TEST_input_64[i, ..., k] = data[..., c]

        if 'pos' in name:
            TEST_target[i] = 1.0
        else:
            TEST_target[i] = 0.0

# Crerate model
#model = mu.create_model_base(input_shape=(64, 64, 15), depths=[3, 3, 27, 3], projection_dims=[32, 64, 96, 128], first_pool=4)
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
        # the size of the weights always match, this will never happen
        ewraewthws

# dump new weights to the model
model.set_weights(W_new)

# compile just in case
# model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.SGD(lr=0))

# predict feature vectors
Y_vector = model.predict([TEST_input_64,])

# Save as numpy file
save_dict = {}
save_dict['y_true'] = TEST_target
save_dict['y_vector'] = Y_vector
save_name = "/glade/work/ksha/NCAR/TRAIN_{}_vec_lead{}_part{}_{}.npy".format(ver, lead, part, prefix)
print(save_name); np.save(save_name, save_dict)



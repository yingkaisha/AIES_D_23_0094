'''
Compute CGAN outputs for the testing set samples.
'''

# general tools
import os
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
args = vars(parser.parse_args())

# =============== #
lead = int(args['lead'])
N_vars = L_vars = 15
# =============== #

N_ens = 49
std_noise = 0.5

weights_round = 5
name_gen_skew = 'GAN_gen{}_skew_wild'.format(weights_round) #False

weights_round = 5
name_gen_norm = 'GAN_gen{}_norm_wild'.format(weights_round) #False

input_flag = [False, False, True, True, True, False, False, True, True, False, False, False, False, False, False]
output_skew_flag = [True, False, False, False, False, False, False, False, False, True, True, False, False, False, False]
output_norm_flag = [False, True, False, False, False, True, True, False, False, False, False, True, True, True, True]

N_micro = sum(input_flag)
N_envir_skew = sum(output_skew_flag)
N_envir_norm = sum(output_norm_flag)

model_gen_skew = mu.GAN_G(N_envir=N_envir_skew, N_micro=N_micro)
W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(name_gen_skew))
model_gen_skew.set_weights(W_old)

model_gen_norm = mu.GAN_G(N_envir=N_envir_norm, N_micro=N_micro)
W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(name_gen_norm))
model_gen_norm.set_weights(W_old)

# Collection batch names from scratch and campaign
names = glob("{}TEST*lead{}.npy".format(path_batch_v4, lead))
filename_valid = sorted(names)
# filename_valid = filename_valid[:100]

print(len(filename_valid))

L_var = L_vars

TEST_input_64 = np.empty((N_ens+1, 64, 64, L_var))
GAN_output = np.empty((N_ens, 64, 64, L_var))

for i, name in enumerate(filename_valid):
    print(i)
    start_time = time.time()
    data = np.load(name, allow_pickle=True)
    
    TEST_input_64[0, ...] = data
    
    # ===== CGAN section ===== # 
    GAN_output[...] = data
    
    input_skew = GAN_output[..., output_skew_flag]
    input_norm = GAN_output[..., output_norm_flag]
    input_ref = GAN_output[..., input_flag]

    noise_skew = np.random.normal(0, std_noise, size=(N_ens, 64, 64, N_envir_skew))
    noise_norm = np.random.normal(0, std_noise, size=(N_ens, 64, 64, N_envir_norm))

    input_skew_ = input_skew+noise_skew
    input_skew_[input_skew_<0] = 0
    
    output_skew = model_gen_skew.predict([input_skew_, input_ref])
    output_skew[output_skew<0] = 0
    GAN_output[..., output_skew_flag] = output_skew
    GAN_output[..., output_norm_flag] = model_gen_norm.predict([input_norm+noise_norm, input_ref])
    
    TEST_input_64[1:, ...] = GAN_output
    name_save = '/glade/derecho/scratch/ksha/{}'.format(os.path.basename(name))
    print(name_save)
    np.save(name_save, TEST_input_64)

    print("--- %s seconds ---" % (time.time() - start_time))

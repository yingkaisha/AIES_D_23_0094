# general tools
import os
import re
import sys
import time
import random
from glob import glob

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('part', help='part')
args = vars(parser.parse_args())
part = int(args['part'])
# =============== #


def id_extract(filenames):
    
    indx_out = []
    indy_out = []
    lead_out = []
    day_out = []
    
    for i, name in enumerate(filenames):        
        nums = re.findall(r'\d+', name)
        lead = int(nums[-1])
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
                
        indx_out.append(indx)
        indy_out.append(indy)
        lead_out.append(lead)
        day_out.append(day)
        
    return np.array(indx_out), np.array(indy_out), np.array(lead_out), np.array(day_out)

# import filenames
filenames = np.load('/glade/work/ksha/NCAR/filenames_for_TRAIN_inds_v4x.npy', allow_pickle=True)[()]

# 2, 3, 4, 5
# 20, 21, 22, 23

# ====================================== #
gap = 40000
rad = 1308*2
# ====================================== #
#21
for lead1_base in range(2, 21):
    
    count = 0
    
    lead2_base = lead1_base + 1
    lead3_base = lead1_base + 2
    lead4_base = lead1_base + 3
    
    filename_train_lead2 = filenames['lead{}'.format(lead1_base)]
    filename_train_lead3 = filenames['lead{}'.format(lead2_base)]
    filename_train_lead4 = filenames['lead{}'.format(lead3_base)]
    filename_train_lead5 = filenames['lead{}'.format(lead4_base)]
    
    indx2, indy2, lead2, day2 = id_extract(filename_train_lead2)
    indx3, indy3, lead3, day3 = id_extract(filename_train_lead3)
    indx4, indy4, lead4, day4 = id_extract(filename_train_lead4)
    indx5, indy5, lead5, day5 = id_extract(filename_train_lead5)

    L = len(filename_train_lead2)

    ind_end = (part+1)*gap
    ind_end_min = np.min([L, ind_end])
    picks = range(part*gap, ind_end_min, 1)

    ind2 = np.empty(L); ind3 = np.empty(L); ind4 = np.empty(L); ind5 = np.empty(L)
    
    start_time = time.time()

    for i in picks:

        i_start = np.max([i-rad, 0])
        i_end = np.min([i+rad, L])

        ind_temp2 = np.nan; ind_temp3 = np.nan; ind_temp4 = np.nan; ind_temp5 = np.nan

        pattern_day = 'TRAIN_day{:03d}'.format(day2[i])

        patten_lead3 = 'indx{}_indy{}_lead{}'.format(indx2[i], indy2[i], lead2_base)
        patten_lead4 = 'indx{}_indy{}_lead{}'.format(indx2[i], indy2[i], lead3_base)
        patten_lead5 = 'indx{}_indy{}_lead{}'.format(indx2[i], indy2[i], lead4_base)

        for i3, name3 in enumerate(filename_train_lead3[i_start:i_end]):
            if (pattern_day in name3) and (patten_lead3 in name3):
                ind_temp3 = i_start+i3
                break;

        for i4, name4 in enumerate(filename_train_lead4[i_start:i_end]):
            if (pattern_day in name4) and (patten_lead4 in name4):
                ind_temp4 = i_start+i4
                break;

        for i5, name5 in enumerate(filename_train_lead5[i_start:i_end]):
            if (pattern_day in name5) and (patten_lead5 in name5):
                ind_temp5 = i_start+i5
                break;

        flag = ind_temp3
        flag = flag + ind_temp4 + ind_temp5

        if np.logical_not(np.isnan(flag)): 
            ind2[count] = i
            ind3[count] = ind_temp3; ind4[count] = ind_temp4; ind5[count] = ind_temp5
            count += 1

    print("--- %s seconds ---" % (time.time() - start_time))

    print(count)
    save_dict = {}
    save_dict['ind{}'.format(lead1_base)] = ind2[:count]
    save_dict['ind{}'.format(lead2_base)] = ind3[:count]
    save_dict['ind{}'.format(lead3_base)] = ind4[:count]
    save_dict['ind{}'.format(lead4_base)] = ind5[:count]

    np.save('/glade/work/ksha/NCAR/TRAIN_lead_inds_{}{}{}{}_part{}_v4x.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base, part), save_dict)
    print('/glade/work/ksha/NCAR/TRAIN_lead_inds_{}{}{}{}_part{}_v4x.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base, part))
    

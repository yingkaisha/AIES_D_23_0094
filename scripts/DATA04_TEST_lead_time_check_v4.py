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

# =============== #
part = int(args['part'])

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

filename_train_lead2 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead2.npy"))
filename_train_lead3 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead3.npy"))
filename_train_lead4 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead4.npy"))
filename_train_lead5 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead5.npy"))
filename_train_lead6 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead6.npy"))
filename_train_lead7 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead7.npy"))
filename_train_lead8 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead8.npy"))
filename_train_lead9 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead9.npy"))
filename_train_lead10 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead10.npy"))
filename_train_lead11 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead11.npy"))
filename_train_lead12 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead12.npy"))
filename_train_lead13 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead13.npy"))
filename_train_lead14 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead14.npy"))
filename_train_lead15 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead15.npy"))
filename_train_lead16 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead16.npy"))
filename_train_lead17 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead17.npy"))
filename_train_lead18 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead18.npy"))
filename_train_lead19 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead19.npy"))
filename_train_lead20 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead20.npy"))
filename_train_lead21 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead21.npy"))
filename_train_lead22 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead22.npy"))
filename_train_lead23 = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/*TEST*lead23.npy"))


print('{}, {}, {}, {}, {}'.format(len(filename_train_lead2), 
                                  len(filename_train_lead3), 
                                  len(filename_train_lead4), 
                                  len(filename_train_lead5), 
                                  len(filename_train_lead6)))

indx2, indy2, lead2, day2 = id_extract(filename_train_lead2)
indx3, indy3, lead3, day3 = id_extract(filename_train_lead3)
indx4, indy4, lead4, day4 = id_extract(filename_train_lead4)
indx5, indy5, lead5, day5 = id_extract(filename_train_lead5)
indx6, indy6, lead6, day6 = id_extract(filename_train_lead6)
indx7, indy7, lead7, day7 = id_extract(filename_train_lead7)
indx8, indy8, lead8, day8 = id_extract(filename_train_lead8)
indx9, indy9, lead9, day9 = id_extract(filename_train_lead9)
indx10, indy10, lead10, day10 = id_extract(filename_train_lead10)
indx11, indy11, lead11, day11 = id_extract(filename_train_lead11)
indx12, indy12, lead12, day12 = id_extract(filename_train_lead12)
indx13, indy13, lead13, day13 = id_extract(filename_train_lead13)
indx14, indy14, lead14, day14 = id_extract(filename_train_lead14)
indx15, indy15, lead15, day15 = id_extract(filename_train_lead15)
indx16, indy16, lead16, day16 = id_extract(filename_train_lead16)
indx17, indy17, lead17, day17 = id_extract(filename_train_lead17)
indx18, indy18, lead18, day18 = id_extract(filename_train_lead18)
indx19, indy19, lead19, day19 = id_extract(filename_train_lead19)
indx20, indy20, lead20, day20 = id_extract(filename_train_lead20)
indx21, indy21, lead21, day21 = id_extract(filename_train_lead21)
indx22, indy22, lead22, day22 = id_extract(filename_train_lead22)
indx23, indy23, lead23, day23 = id_extract(filename_train_lead23)

L = len(filename_train_lead2)

ind2 = np.empty(L); ind3 = np.empty(L); ind4 = np.empty(L); ind5 = np.empty(L)
ind6 = np.empty(L); ind7 = np.empty(L); ind8 = np.empty(L); ind9 = np.empty(L); 
ind10 = np.empty(L); ind11 = np.empty(L); ind12 = np.empty(L); ind13 = np.empty(L); 
ind14 = np.empty(L); ind15 = np.empty(L); ind16 = np.empty(L); ind17 = np.empty(L); 
ind18 = np.empty(L); ind19 = np.empty(L); ind20 = np.empty(L); ind21 = np.empty(L); 
ind22 = np.empty(L); ind23 = np.empty(L)

count = 0

start_time = time.time()

gap = 40000
ind_end = (part+1)*gap
ind_end_min = np.min([L, ind_end])

picks = range(part*gap, ind_end_min, 1)

rad = 1308*3

for i in picks:

    i_start = np.max([i-rad, 0])
    i_end = np.min([i+rad, L])
    
    ind_temp2 = np.nan; ind_temp3 = np.nan; ind_temp4 = np.nan
    ind_temp5 = np.nan; ind_temp6 = np.nan; ind_temp7 = np.nan; 
    ind_temp8 = np.nan; ind_temp9 = np.nan; ind_temp10 = np.nan; 
    ind_temp11 = np.nan; ind_temp12 = np.nan; ind_temp13 = np.nan; 
    ind_temp14 = np.nan; ind_temp15 = np.nan; ind_temp16 = np.nan; 
    ind_temp17 = np.nan; ind_temp18 = np.nan; ind_temp19 = np.nan; 
    ind_temp20 = np.nan; ind_temp21 = np.nan; ind_temp22 = np.nan; 
    ind_temp23 = np.nan

    pattern_day = 'TESTv4_day{:03d}'.format(day2[i])

    patten_lead3 = 'indx{}_indy{}_lead3'.format(indx2[i], indy2[i])
    patten_lead4 = 'indx{}_indy{}_lead4'.format(indx2[i], indy2[i])
    patten_lead5 = 'indx{}_indy{}_lead5'.format(indx2[i], indy2[i])
    patten_lead6 = 'indx{}_indy{}_lead6'.format(indx2[i], indy2[i])
    patten_lead7 = 'indx{}_indy{}_lead7'.format(indx2[i], indy2[i])
    patten_lead8 = 'indx{}_indy{}_lead8'.format(indx2[i], indy2[i])
    patten_lead9 = 'indx{}_indy{}_lead9'.format(indx2[i], indy2[i])
    patten_lead10 = 'indx{}_indy{}_lead10'.format(indx2[i], indy2[i])
    patten_lead11 = 'indx{}_indy{}_lead11'.format(indx2[i], indy2[i])
    patten_lead12 = 'indx{}_indy{}_lead12'.format(indx2[i], indy2[i])
    patten_lead13 = 'indx{}_indy{}_lead13'.format(indx2[i], indy2[i])
    patten_lead14 = 'indx{}_indy{}_lead14'.format(indx2[i], indy2[i])
    patten_lead15 = 'indx{}_indy{}_lead15'.format(indx2[i], indy2[i])
    patten_lead16 = 'indx{}_indy{}_lead16'.format(indx2[i], indy2[i])
    patten_lead17 = 'indx{}_indy{}_lead17'.format(indx2[i], indy2[i])
    patten_lead18 = 'indx{}_indy{}_lead18'.format(indx2[i], indy2[i])
    patten_lead19 = 'indx{}_indy{}_lead19'.format(indx2[i], indy2[i])
    patten_lead20 = 'indx{}_indy{}_lead20'.format(indx2[i], indy2[i])
    patten_lead21 = 'indx{}_indy{}_lead21'.format(indx2[i], indy2[i])
    patten_lead22 = 'indx{}_indy{}_lead22'.format(indx2[i], indy2[i])
    patten_lead23 = 'indx{}_indy{}_lead23'.format(indx2[i], indy2[i])

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

    for i6, name6 in enumerate(filename_train_lead6[i_start:i_end]):
        if (pattern_day in name6) and (patten_lead6 in name6):
            ind_temp6 = i_start+i6
            break;
          
    for i7, name7 in enumerate(filename_train_lead7[i_start:i_end]):
        if (pattern_day in name7) and (patten_lead7 in name7):
            ind_temp7 = i_start+i7
            break;
            
    for i8, name8 in enumerate(filename_train_lead8[i_start:i_end]):
        if (pattern_day in name8) and (patten_lead8 in name8):
            ind_temp8 = i_start+i8
            break;
            
    for i9, name9 in enumerate(filename_train_lead9[i_start:i_end]):
        if (pattern_day in name9) and (patten_lead9 in name9):
            ind_temp9 = i_start+i9
            break;
            
    for i10, name10 in enumerate(filename_train_lead10[i_start:i_end]):
        if (pattern_day in name10) and (patten_lead10 in name10):
            ind_temp10 = i_start+i10
            break;
            
    for i11, name11 in enumerate(filename_train_lead11[i_start:i_end]):
        if (pattern_day in name11) and (patten_lead11 in name11):
            ind_temp11 = i_start+i11
            break;
            
    for i12, name12 in enumerate(filename_train_lead12[i_start:i_end]):
        if (pattern_day in name12) and (patten_lead12 in name12):
            ind_temp12 = i_start+i12
            break;
            
    for i12, name12 in enumerate(filename_train_lead12[i_start:i_end]):
        if (pattern_day in name12) and (patten_lead12 in name12):
            ind_temp12 = i_start+i12
            break;
            
    for i13, name13 in enumerate(filename_train_lead13[i_start:i_end]):
        if (pattern_day in name13) and (patten_lead13 in name13):
            ind_temp13 = i_start+i13
            break;
        
    for i14, name14 in enumerate(filename_train_lead14[i_start:i_end]):
        if (pattern_day in name14) and (patten_lead14 in name14):
            ind_temp14 = i_start+i14
            break;
            
    for i15, name15 in enumerate(filename_train_lead15[i_start:i_end]):
        if (pattern_day in name15) and (patten_lead15 in name15):
            ind_temp15 = i_start+i15
            break;
            
    for i16, name16 in enumerate(filename_train_lead16[i_start:i_end]):
        if (pattern_day in name16) and (patten_lead16 in name16):
            ind_temp16 = i_start+i16
            break;
            
    for i17, name17 in enumerate(filename_train_lead17[i_start:i_end]):
        if (pattern_day in name17) and (patten_lead17 in name17):
            ind_temp17 = i_start+i17
            break;
            
    for i18, name18 in enumerate(filename_train_lead18[i_start:i_end]):
        if (pattern_day in name18) and (patten_lead18 in name18):
            ind_temp18 = i_start+i18
            break;
            
    for i19, name19 in enumerate(filename_train_lead19[i_start:i_end]):
        if (pattern_day in name19) and (patten_lead19 in name19):
            ind_temp19 = i_start+i19
            break;
        
    for i20, name20 in enumerate(filename_train_lead20[i_start:i_end]):
        if (pattern_day in name20) and (patten_lead20 in name20):
            ind_temp20 = i_start+i20
            break;
            
    for i21, name21 in enumerate(filename_train_lead21[i_start:i_end]):
        if (pattern_day in name21) and (patten_lead21 in name21):
            ind_temp21 = i_start+i21
            break;
            
    for i22, name22 in enumerate(filename_train_lead22[i_start:i_end]):
        if (pattern_day in name22) and (patten_lead22 in name22):
            ind_temp22 = i_start+i22
            break;
            
    for i23, name23 in enumerate(filename_train_lead23[i_start:i_end]):
        if (pattern_day in name23) and (patten_lead23 in name23):
            ind_temp23 = i_start+i23
            break;

    flag = ind_temp3
    flag = flag + ind_temp4 + ind_temp5 + ind_temp6
    flag = flag + ind_temp7 + ind_temp8 + ind_temp9
    flag = flag + ind_temp10 + ind_temp11 + ind_temp12
    flag = flag + ind_temp13 + ind_temp14 + ind_temp15
    flag = flag + ind_temp16 + ind_temp17 + ind_temp18
    flag = flag + ind_temp19 + ind_temp20 + ind_temp21
    flag = flag + ind_temp22 + ind_temp23

    if np.logical_not(np.isnan(flag)): 
        ind2[count] = i
        ind3[count] = ind_temp3; ind4[count] = ind_temp4; ind5[count] = ind_temp5
        ind6[count] = ind_temp6; ind7[count] = ind_temp7; ind8[count] = ind_temp8; 
        ind9[count] = ind_temp9; ind10[count] = ind_temp10; ind11[count] = ind_temp11; 
        ind12[count] = ind_temp12; ind13[count] = ind_temp13; ind14[count] = ind_temp14; 
        ind15[count] = ind_temp15; ind16[count] = ind_temp16; ind17[count] = ind_temp17; 
        ind18[count] = ind_temp18; ind19[count] = ind_temp19; ind20[count] = ind_temp20; 
        ind21[count] = ind_temp21; ind22[count] = ind_temp22; ind23[count] = ind_temp23
        count += 1

print("--- %s seconds ---" % (time.time() - start_time))

print(count)
save_dict = {}
save_dict['ind2'] = ind2[:count]
save_dict['ind3'] = ind3[:count]
save_dict['ind4'] = ind4[:count]
save_dict['ind5'] = ind5[:count]
save_dict['ind6'] = ind6[:count]

save_dict['ind7'] = ind7[:count]
save_dict['ind8'] = ind8[:count]
save_dict['ind9'] = ind9[:count]
save_dict['ind10'] = ind10[:count]
save_dict['ind11'] = ind11[:count]
save_dict['ind12'] = ind12[:count]
save_dict['ind13'] = ind13[:count]
save_dict['ind14'] = ind14[:count]
save_dict['ind15'] = ind15[:count]
save_dict['ind16'] = ind16[:count]
save_dict['ind17'] = ind17[:count]
save_dict['ind18'] = ind18[:count]
save_dict['ind19'] = ind19[:count]

save_dict['ind20'] = ind20[:count]
save_dict['ind21'] = ind21[:count]
save_dict['ind22'] = ind22[:count]
save_dict['ind23'] = ind23[:count]


np.save('/glade/work/ksha/NCAR/TEST_lead_inds_v4_part{}.npy'.format(part), save_dict)
print('/glade/work/ksha/NCAR/TEST_lead_inds_v4_part{}.npy'.format(part))
    

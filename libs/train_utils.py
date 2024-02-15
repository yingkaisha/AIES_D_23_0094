import re
import os
import random
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def name_extract(filenames):
    '''Separate train, valid, and test patches based on their filenames.
      Returns 2 lists of filenames: filename_train, filename_valid'''
    
    date_train_end = datetime(2020, 7, 14)
    date_valid_end = datetime(2021, 1, 1)
    
    filename_train = []
    filename_valid = []
    
    # --------------------------------------- #
    base_v3_s = datetime(2018, 7, 15)
    base_v4_s = datetime(2020, 12, 3)
    base_v4x_s = datetime(2019, 10, 1)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+365+30)]
    date_list_v4x = [base_v4x_s + timedelta(days=day) for day in range(429)]
    # ---------------------------------------- #
    
    for i, name in enumerate(filenames):
        
        if 'v4x' in name:
            date_list = date_list_v4x
        elif 'v4' in name:
            date_list = date_list_v4
        else:
            date_list = date_list_v3
        
        nums = re.findall(r'\d+', name)
        day = int(nums[-4])
        day = date_list[day]
        
        if (day - date_train_end).days < 0:
            filename_train.append(name)
            
        else:
            if (day - date_valid_end).days < 0:
                filename_valid.append(name)

    return filename_train, filename_valid
    
    
def feature_extract(filenames, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max):
    
    lon_out = []
    lat_out = []
    elev_out = []
    mon_out = []
    
    base_v3_s = datetime(2018, 7, 15)
    base_v3_e = datetime(2020, 12, 2)

    base_v4_s = datetime(2020, 12, 3)
    base_v4_e = datetime(2022, 7, 15)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180-151)]
    
    for i, name in enumerate(filenames):
        
        if 'v4' in name:
            date_list = date_list_v4
        else:
            date_list = date_list_v3
        
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        day = date_list[day]
        month = day.month
        
        month_norm = (month - 1)/(12-1)
        
        lon = lon_80km[indx, indy]
        lat = lat_80km[indx, indy]

        lon = (lon - lon_minmax[0])/(lon_minmax[1] - lon_minmax[0])
        lat = (lat - lat_minmax[0])/(lat_minmax[1] - lat_minmax[0])

        elev = elev_80km[indx, indy]
        elev = elev / elev_max
        
        lon_out.append(lon)
        lat_out.append(lat)
        elev_out.append(elev)
        mon_out.append(month_norm)
        
    return np.array(lon_out), np.array(lat_out), np.array(elev_out), np.array(mon_out)
    
    
def name_to_ind(filenames):
    
    indx_out = []
    indy_out = []
    day_out = []
    flag_out = []
    
    for i, name in enumerate(filenames):
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        
        indx_out.append(indx)
        indy_out.append(indy)
        day_out.append(day)
        
        if "pos" in name:
            flag_out.append(True)
        else:
            flag_out.append(False)
        
    return np.array(indx_out), np.array(indy_out), np.array(day_out), np.array(flag_out)

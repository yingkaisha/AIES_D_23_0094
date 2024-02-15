
import numpy as np

prefix = '/glade/work/ksha/NCAR/TRAIN_lead_inds_v4x_part{}.npy'

TRAIN_inds_p0  = np.load(prefix.format(0), allow_pickle=True)[()]
TRAIN_inds_p1  = np.load(prefix.format(1), allow_pickle=True)[()]
TRAIN_inds_p2  = np.load(prefix.format(2), allow_pickle=True)[()]
TRAIN_inds_p3  = np.load(prefix.format(3), allow_pickle=True)[()]

TRAIN_ind2 = np.concatenate((TRAIN_inds_p0['ind2'], TRAIN_inds_p1['ind2'], TRAIN_inds_p2['ind2'], TRAIN_inds_p3['ind2']))

TRAIN_ind3 = np.concatenate((TRAIN_inds_p0['ind3'], TRAIN_inds_p1['ind3'], TRAIN_inds_p2['ind3'], TRAIN_inds_p3['ind3']))

TRAIN_ind4 = np.concatenate((TRAIN_inds_p0['ind4'], TRAIN_inds_p1['ind4'], TRAIN_inds_p2['ind4'], TRAIN_inds_p3['ind4']))

TRAIN_ind5 = np.concatenate((TRAIN_inds_p0['ind5'], TRAIN_inds_p1['ind5'], TRAIN_inds_p2['ind5'], TRAIN_inds_p3['ind5']))

TRAIN_ind6 = np.concatenate((TRAIN_inds_p0['ind6'], TRAIN_inds_p1['ind6'], TRAIN_inds_p2['ind6'], TRAIN_inds_p3['ind6']))

IND_TRAIN_lead = {}
IND_TRAIN_lead['lead2'] = TRAIN_ind2
IND_TRAIN_lead['lead3'] = TRAIN_ind3
IND_TRAIN_lead['lead4'] = TRAIN_ind4
IND_TRAIN_lead['lead5'] = TRAIN_ind5
IND_TRAIN_lead['lead6'] = TRAIN_ind6

prefix = '/glade/work/ksha/NCAR/VALID_lead_inds_v4x_part{}.npy'

VALID_inds_p0 = np.load(prefix.format(0), allow_pickle=True)[()]
VALID_inds_p1 = np.load(prefix.format(1), allow_pickle=True)[()]
VALID_inds_p2 = np.load(prefix.format(2), allow_pickle=True)[()]
VALID_inds_p3 = np.load(prefix.format(3), allow_pickle=True)[()]
VALID_inds_p4 = np.load(prefix.format(4), allow_pickle=True)[()]
VALID_inds_p5 = np.load(prefix.format(5), allow_pickle=True)[()]
VALID_inds_p6 = np.load(prefix.format(6), allow_pickle=True)[()]

VALID_ind2 = np.concatenate((VALID_inds_p0['ind2'], VALID_inds_p1['ind2'], VALID_inds_p2['ind2'],
                             VALID_inds_p3['ind2'], VALID_inds_p4['ind2'], VALID_inds_p5['ind2'], 
                             VALID_inds_p6['ind2']))

VALID_ind3 = np.concatenate((VALID_inds_p0['ind3'], VALID_inds_p1['ind3'], VALID_inds_p2['ind3'],
                             VALID_inds_p3['ind3'], VALID_inds_p4['ind3'], VALID_inds_p5['ind3'], 
                             VALID_inds_p6['ind3']))

VALID_ind4 = np.concatenate((VALID_inds_p0['ind4'], VALID_inds_p1['ind4'], VALID_inds_p2['ind4'],
                             VALID_inds_p3['ind4'], VALID_inds_p4['ind4'], VALID_inds_p5['ind4'], 
                             VALID_inds_p6['ind4']))

VALID_ind5 = np.concatenate((VALID_inds_p0['ind5'], VALID_inds_p1['ind5'], VALID_inds_p2['ind5'],
                             VALID_inds_p3['ind5'], VALID_inds_p4['ind5'], VALID_inds_p5['ind5'], 
                             VALID_inds_p6['ind5']))

VALID_ind6 = np.concatenate((VALID_inds_p0['ind6'], VALID_inds_p1['ind6'], VALID_inds_p2['ind6'],
                             VALID_inds_p3['ind6'], VALID_inds_p4['ind6'], VALID_inds_p5['ind6'], 
                             VALID_inds_p6['ind6']))

IND_VALID_lead = {}
IND_VALID_lead['lead2'] = VALID_ind2
IND_VALID_lead['lead3'] = VALID_ind3
IND_VALID_lead['lead4'] = VALID_ind4
IND_VALID_lead['lead5'] = VALID_ind5
IND_VALID_lead['lead6'] = VALID_ind6

np.save('/glade/work/ksha/NCAR/IND_TRAIN_lead_v4x.npy', IND_TRAIN_lead)
print('/glade/work/ksha/NCAR/IND_TRAIN_lead_full.npy')

np.save('/glade/work/ksha/NCAR/IND_VALID_lead_v4x.npy', IND_VALID_lead)
print('/glade/work/ksha/NCAR/IND_VALID_lead_full.npy')


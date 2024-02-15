
import numpy as np

prefix = '/glade/work/ksha/NCAR/TEST_lead_inds_v4_part{}.npy'

TRAIN_inds_p0  = np.load(prefix.format(0), allow_pickle=True)[()]
TRAIN_inds_p1  = np.load(prefix.format(1), allow_pickle=True)[()]
TRAIN_inds_p2  = np.load(prefix.format(2), allow_pickle=True)[()]
TRAIN_inds_p3  = np.load(prefix.format(3), allow_pickle=True)[()]
TRAIN_inds_p4  = np.load(prefix.format(4), allow_pickle=True)[()]
TRAIN_inds_p5  = np.load(prefix.format(5), allow_pickle=True)[()]
TRAIN_inds_p6  = np.load(prefix.format(6), allow_pickle=True)[()]
TRAIN_inds_p7  = np.load(prefix.format(7), allow_pickle=True)[()]
TRAIN_inds_p8  = np.load(prefix.format(8), allow_pickle=True)[()]
TRAIN_inds_p9  = np.load(prefix.format(9), allow_pickle=True)[()]
TRAIN_inds_p10 = np.load(prefix.format(10), allow_pickle=True)[()]
TRAIN_inds_p11 = np.load(prefix.format(11), allow_pickle=True)[()]


TRAIN_ind2 = np.concatenate((TRAIN_inds_p0['ind2'], TRAIN_inds_p1['ind2'], TRAIN_inds_p2['ind2'], TRAIN_inds_p3['ind2'],
                             TRAIN_inds_p4['ind2'], TRAIN_inds_p5['ind2'], TRAIN_inds_p6['ind2'], TRAIN_inds_p7['ind2'],
                             TRAIN_inds_p8['ind2'], TRAIN_inds_p9['ind2'], TRAIN_inds_p10['ind2'],
                             TRAIN_inds_p11['ind2']))

TRAIN_ind3 = np.concatenate((TRAIN_inds_p0['ind3'], TRAIN_inds_p1['ind3'], TRAIN_inds_p2['ind3'], TRAIN_inds_p3['ind3'],
                             TRAIN_inds_p4['ind3'], TRAIN_inds_p5['ind3'], TRAIN_inds_p6['ind3'], TRAIN_inds_p7['ind3'],
                             TRAIN_inds_p8['ind3'], TRAIN_inds_p9['ind3'], TRAIN_inds_p10['ind3'],
                             TRAIN_inds_p11['ind3']))

TRAIN_ind4 = np.concatenate((TRAIN_inds_p0['ind4'], TRAIN_inds_p1['ind4'], TRAIN_inds_p2['ind4'], TRAIN_inds_p3['ind4'],
                             TRAIN_inds_p4['ind4'], TRAIN_inds_p5['ind4'], TRAIN_inds_p6['ind4'], TRAIN_inds_p7['ind4'],
                             TRAIN_inds_p8['ind4'], TRAIN_inds_p9['ind4'], TRAIN_inds_p10['ind4'],
                             TRAIN_inds_p11['ind4']))

TRAIN_ind5 = np.concatenate((TRAIN_inds_p0['ind5'], TRAIN_inds_p1['ind5'], TRAIN_inds_p2['ind5'], TRAIN_inds_p3['ind5'],
                             TRAIN_inds_p4['ind5'], TRAIN_inds_p5['ind5'], TRAIN_inds_p6['ind5'], TRAIN_inds_p7['ind5'],
                             TRAIN_inds_p8['ind5'], TRAIN_inds_p9['ind5'], TRAIN_inds_p10['ind5'],
                             TRAIN_inds_p11['ind5']))

TRAIN_ind6 = np.concatenate((TRAIN_inds_p0['ind6'], TRAIN_inds_p1['ind6'], TRAIN_inds_p2['ind6'], TRAIN_inds_p3['ind6'],
                             TRAIN_inds_p4['ind6'], TRAIN_inds_p5['ind6'], TRAIN_inds_p6['ind6'], TRAIN_inds_p7['ind6'],
                             TRAIN_inds_p8['ind6'], TRAIN_inds_p9['ind6'], TRAIN_inds_p10['ind6'],
                             TRAIN_inds_p11['ind6']))

TRAIN_ind7 = np.concatenate((TRAIN_inds_p0['ind7'], TRAIN_inds_p1['ind7'], TRAIN_inds_p2['ind7'], TRAIN_inds_p3['ind7'],
                             TRAIN_inds_p4['ind7'], TRAIN_inds_p5['ind7'], TRAIN_inds_p6['ind7'], TRAIN_inds_p7['ind7'],
                             TRAIN_inds_p8['ind7'], TRAIN_inds_p9['ind7'], TRAIN_inds_p10['ind7'],
                             TRAIN_inds_p11['ind7']))

TRAIN_ind8 = np.concatenate((TRAIN_inds_p0['ind8'], TRAIN_inds_p1['ind8'], TRAIN_inds_p2['ind8'], TRAIN_inds_p3['ind8'],
                             TRAIN_inds_p4['ind8'], TRAIN_inds_p5['ind8'], TRAIN_inds_p6['ind8'], TRAIN_inds_p7['ind8'],
                             TRAIN_inds_p8['ind8'], TRAIN_inds_p9['ind8'], TRAIN_inds_p10['ind8'],
                             TRAIN_inds_p11['ind8']))


TRAIN_ind9 = np.concatenate((TRAIN_inds_p0['ind9'], TRAIN_inds_p1['ind9'], TRAIN_inds_p2['ind9'], TRAIN_inds_p3['ind9'],
                              TRAIN_inds_p4['ind9'], TRAIN_inds_p5['ind9'], TRAIN_inds_p6['ind9'], TRAIN_inds_p7['ind9'],
                              TRAIN_inds_p8['ind9'], TRAIN_inds_p9['ind9'], TRAIN_inds_p10['ind9'],
                              TRAIN_inds_p11['ind9']))

TRAIN_ind10 = np.concatenate((TRAIN_inds_p0['ind10'], TRAIN_inds_p1['ind10'], TRAIN_inds_p2['ind10'], TRAIN_inds_p3['ind10'],
                              TRAIN_inds_p4['ind10'], TRAIN_inds_p5['ind10'], TRAIN_inds_p6['ind10'], TRAIN_inds_p7['ind10'],
                              TRAIN_inds_p8['ind10'], TRAIN_inds_p9['ind10'], TRAIN_inds_p10['ind10'],
                              TRAIN_inds_p11['ind10']))

TRAIN_ind11 = np.concatenate((TRAIN_inds_p0['ind11'], TRAIN_inds_p1['ind11'], TRAIN_inds_p2['ind11'], TRAIN_inds_p3['ind11'],
                              TRAIN_inds_p4['ind11'], TRAIN_inds_p5['ind11'], TRAIN_inds_p6['ind11'], TRAIN_inds_p7['ind11'],
                              TRAIN_inds_p8['ind11'], TRAIN_inds_p9['ind11'], TRAIN_inds_p10['ind11'],
                              TRAIN_inds_p11['ind11']))

TRAIN_ind12 = np.concatenate((TRAIN_inds_p0['ind12'], TRAIN_inds_p1['ind12'], TRAIN_inds_p2['ind12'], TRAIN_inds_p3['ind12'],
                              TRAIN_inds_p4['ind12'], TRAIN_inds_p5['ind12'], TRAIN_inds_p6['ind12'], TRAIN_inds_p7['ind12'],
                              TRAIN_inds_p8['ind12'], TRAIN_inds_p9['ind12'], TRAIN_inds_p10['ind12'],
                              TRAIN_inds_p11['ind12']))

TRAIN_ind13 = np.concatenate((TRAIN_inds_p0['ind13'], TRAIN_inds_p1['ind13'], TRAIN_inds_p2['ind13'], TRAIN_inds_p3['ind13'],
                              TRAIN_inds_p4['ind13'], TRAIN_inds_p5['ind13'], TRAIN_inds_p6['ind13'], TRAIN_inds_p7['ind13'],
                              TRAIN_inds_p8['ind13'], TRAIN_inds_p9['ind13'], TRAIN_inds_p10['ind13'],
                              TRAIN_inds_p11['ind13']))

TRAIN_ind14 = np.concatenate((TRAIN_inds_p0['ind14'], TRAIN_inds_p1['ind14'], TRAIN_inds_p2['ind14'], TRAIN_inds_p3['ind14'],
                              TRAIN_inds_p4['ind14'], TRAIN_inds_p5['ind14'], TRAIN_inds_p6['ind14'], TRAIN_inds_p7['ind14'],
                              TRAIN_inds_p8['ind14'], TRAIN_inds_p9['ind14'], TRAIN_inds_p10['ind14'],
                              TRAIN_inds_p11['ind14']))

TRAIN_ind15 = np.concatenate((TRAIN_inds_p0['ind15'], TRAIN_inds_p1['ind15'], TRAIN_inds_p2['ind15'], TRAIN_inds_p3['ind15'],
                              TRAIN_inds_p4['ind15'], TRAIN_inds_p5['ind15'], TRAIN_inds_p6['ind15'], TRAIN_inds_p7['ind15'],
                              TRAIN_inds_p8['ind15'], TRAIN_inds_p9['ind15'], TRAIN_inds_p10['ind15'],
                              TRAIN_inds_p11['ind15']))

TRAIN_ind16 = np.concatenate((TRAIN_inds_p0['ind16'], TRAIN_inds_p1['ind16'], TRAIN_inds_p2['ind16'], TRAIN_inds_p3['ind16'],
                              TRAIN_inds_p4['ind16'], TRAIN_inds_p5['ind16'], TRAIN_inds_p6['ind16'], TRAIN_inds_p7['ind16'],
                              TRAIN_inds_p8['ind16'], TRAIN_inds_p9['ind16'], TRAIN_inds_p10['ind16'],
                              TRAIN_inds_p11['ind16']))

TRAIN_ind17 = np.concatenate((TRAIN_inds_p0['ind17'], TRAIN_inds_p1['ind17'], TRAIN_inds_p2['ind17'], TRAIN_inds_p3['ind17'],
                              TRAIN_inds_p4['ind17'], TRAIN_inds_p5['ind17'], TRAIN_inds_p6['ind17'], TRAIN_inds_p7['ind17'],
                              TRAIN_inds_p8['ind17'], TRAIN_inds_p9['ind17'], TRAIN_inds_p10['ind17'],
                              TRAIN_inds_p11['ind17']))

TRAIN_ind18 = np.concatenate((TRAIN_inds_p0['ind18'], TRAIN_inds_p1['ind18'], TRAIN_inds_p2['ind18'], TRAIN_inds_p3['ind18'],
                              TRAIN_inds_p4['ind18'], TRAIN_inds_p5['ind18'], TRAIN_inds_p6['ind18'], TRAIN_inds_p7['ind18'],
                              TRAIN_inds_p8['ind18'], TRAIN_inds_p9['ind18'], TRAIN_inds_p10['ind18'],
                              TRAIN_inds_p11['ind18']))

TRAIN_ind19 = np.concatenate((TRAIN_inds_p0['ind19'], TRAIN_inds_p1['ind19'], TRAIN_inds_p2['ind19'], TRAIN_inds_p3['ind19'],
                              TRAIN_inds_p4['ind19'], TRAIN_inds_p5['ind19'], TRAIN_inds_p6['ind19'], TRAIN_inds_p7['ind19'],
                              TRAIN_inds_p8['ind19'], TRAIN_inds_p9['ind19'], TRAIN_inds_p10['ind19'],
                              TRAIN_inds_p11['ind19']))

TRAIN_ind20 = np.concatenate((TRAIN_inds_p0['ind20'], TRAIN_inds_p1['ind20'], TRAIN_inds_p2['ind20'], TRAIN_inds_p3['ind20'],
                              TRAIN_inds_p4['ind20'], TRAIN_inds_p5['ind20'], TRAIN_inds_p6['ind20'], TRAIN_inds_p7['ind20'],
                              TRAIN_inds_p8['ind20'], TRAIN_inds_p9['ind20'], TRAIN_inds_p10['ind20'],
                              TRAIN_inds_p11['ind20']))

TRAIN_ind21 = np.concatenate((TRAIN_inds_p0['ind21'], TRAIN_inds_p1['ind21'], TRAIN_inds_p2['ind21'], TRAIN_inds_p3['ind21'],
                              TRAIN_inds_p4['ind21'], TRAIN_inds_p5['ind21'], TRAIN_inds_p6['ind21'], TRAIN_inds_p7['ind21'],
                              TRAIN_inds_p8['ind21'], TRAIN_inds_p9['ind21'], TRAIN_inds_p10['ind21'],
                              TRAIN_inds_p11['ind21']))

TRAIN_ind22 = np.concatenate((TRAIN_inds_p0['ind22'], TRAIN_inds_p1['ind22'], TRAIN_inds_p2['ind22'], TRAIN_inds_p3['ind22'],
                              TRAIN_inds_p4['ind22'], TRAIN_inds_p5['ind22'], TRAIN_inds_p6['ind22'], TRAIN_inds_p7['ind22'],
                              TRAIN_inds_p8['ind22'], TRAIN_inds_p9['ind22'], TRAIN_inds_p10['ind22'],
                              TRAIN_inds_p11['ind22']))

TRAIN_ind23 = np.concatenate((TRAIN_inds_p0['ind23'], TRAIN_inds_p1['ind23'], TRAIN_inds_p2['ind23'], TRAIN_inds_p3['ind23'],
                              TRAIN_inds_p4['ind23'], TRAIN_inds_p5['ind23'], TRAIN_inds_p6['ind23'], TRAIN_inds_p7['ind23'],
                              TRAIN_inds_p8['ind23'], TRAIN_inds_p9['ind23'], TRAIN_inds_p10['ind23'],
                              TRAIN_inds_p11['ind23']))

IND_TRAIN_lead = {}

IND_TRAIN_lead['lead2'] = TRAIN_ind2
IND_TRAIN_lead['lead3'] = TRAIN_ind3
IND_TRAIN_lead['lead4'] = TRAIN_ind4
IND_TRAIN_lead['lead5'] = TRAIN_ind5
IND_TRAIN_lead['lead6'] = TRAIN_ind6
IND_TRAIN_lead['lead7'] = TRAIN_ind7
IND_TRAIN_lead['lead8'] = TRAIN_ind8

IND_TRAIN_lead['lead9'] = TRAIN_ind9
IND_TRAIN_lead['lead10'] = TRAIN_ind10
IND_TRAIN_lead['lead11'] = TRAIN_ind11
IND_TRAIN_lead['lead12'] = TRAIN_ind12
IND_TRAIN_lead['lead13'] = TRAIN_ind13
IND_TRAIN_lead['lead14'] = TRAIN_ind14
IND_TRAIN_lead['lead15'] = TRAIN_ind15
IND_TRAIN_lead['lead16'] = TRAIN_ind16
IND_TRAIN_lead['lead17'] = TRAIN_ind17
IND_TRAIN_lead['lead18'] = TRAIN_ind18

IND_TRAIN_lead['lead19'] = TRAIN_ind19
IND_TRAIN_lead['lead20'] = TRAIN_ind20
IND_TRAIN_lead['lead21'] = TRAIN_ind21
IND_TRAIN_lead['lead22'] = TRAIN_ind22
IND_TRAIN_lead['lead23'] = TRAIN_ind23

np.save('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy', IND_TRAIN_lead)
print('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy')

# np.save('/glade/work/ksha/NCAR/IND_VALID_lead_v4x.npy', IND_VALID_lead)
# print('/glade/work/ksha/NCAR/IND_VALID_lead_full.npy')


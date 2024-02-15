
import numpy as np

prefix = '/glade/work/ksha/NCAR/TRAIN_lead_inds_full_part{}.npy'

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
TRAIN_inds_p12 = np.load(prefix.format(12), allow_pickle=True)[()]
TRAIN_inds_p13 = np.load(prefix.format(13), allow_pickle=True)[()]
TRAIN_inds_p14 = np.load(prefix.format(14), allow_pickle=True)[()]
TRAIN_inds_p15 = np.load(prefix.format(15), allow_pickle=True)[()]
TRAIN_inds_p16 = np.load(prefix.format(16), allow_pickle=True)[()]
TRAIN_inds_p17 = np.load(prefix.format(17), allow_pickle=True)[()]
TRAIN_inds_p18 = np.load(prefix.format(18), allow_pickle=True)[()]
TRAIN_inds_p19 = np.load(prefix.format(19), allow_pickle=True)[()]
TRAIN_inds_p20 = np.load(prefix.format(20), allow_pickle=True)[()]



TRAIN_ind2 = np.concatenate((TRAIN_inds_p0['ind2'], TRAIN_inds_p1['ind2'], TRAIN_inds_p2['ind2'], TRAIN_inds_p3['ind2'],
                             TRAIN_inds_p4['ind2'], TRAIN_inds_p5['ind2'], TRAIN_inds_p6['ind2'], TRAIN_inds_p7['ind2'],
                             TRAIN_inds_p8['ind2'], TRAIN_inds_p9['ind2'], TRAIN_inds_p10['ind2'],
                             TRAIN_inds_p11['ind2'], TRAIN_inds_p12['ind2'], TRAIN_inds_p13['ind2'],
                             TRAIN_inds_p14['ind2'], TRAIN_inds_p15['ind2'], TRAIN_inds_p16['ind2'],
                             TRAIN_inds_p17['ind2'], TRAIN_inds_p18['ind2'], TRAIN_inds_p19['ind2'],
                             TRAIN_inds_p20['ind2'],))

TRAIN_ind3 = np.concatenate((TRAIN_inds_p0['ind3'], TRAIN_inds_p1['ind3'], TRAIN_inds_p2['ind3'], TRAIN_inds_p3['ind3'],
                             TRAIN_inds_p4['ind3'], TRAIN_inds_p5['ind3'], TRAIN_inds_p6['ind3'], TRAIN_inds_p7['ind3'],
                             TRAIN_inds_p8['ind3'], TRAIN_inds_p9['ind3'], TRAIN_inds_p10['ind3'],
                             TRAIN_inds_p11['ind3'], TRAIN_inds_p12['ind3'], TRAIN_inds_p13['ind3'],
                             TRAIN_inds_p14['ind3'], TRAIN_inds_p15['ind3'], TRAIN_inds_p16['ind3'],                                                                        TRAIN_inds_p17['ind3'], TRAIN_inds_p18['ind3'], TRAIN_inds_p19['ind3'],
                             TRAIN_inds_p20['ind3'],))

TRAIN_ind4 = np.concatenate((TRAIN_inds_p0['ind4'], TRAIN_inds_p1['ind4'], TRAIN_inds_p2['ind4'], TRAIN_inds_p3['ind4'],
                             TRAIN_inds_p4['ind4'], TRAIN_inds_p5['ind4'], TRAIN_inds_p6['ind4'], TRAIN_inds_p7['ind4'],
                             TRAIN_inds_p8['ind4'], TRAIN_inds_p9['ind4'], TRAIN_inds_p10['ind4'],
                             TRAIN_inds_p11['ind4'], TRAIN_inds_p12['ind4'], TRAIN_inds_p13['ind4'],
                             TRAIN_inds_p14['ind4'], TRAIN_inds_p15['ind4'], TRAIN_inds_p16['ind4'],                                                                        TRAIN_inds_p17['ind4'], TRAIN_inds_p18['ind4'], TRAIN_inds_p19['ind4'],
                             TRAIN_inds_p20['ind4'],))

TRAIN_ind5 = np.concatenate((TRAIN_inds_p0['ind5'], TRAIN_inds_p1['ind5'], TRAIN_inds_p2['ind5'], TRAIN_inds_p3['ind5'],
                             TRAIN_inds_p4['ind5'], TRAIN_inds_p5['ind5'], TRAIN_inds_p6['ind5'], TRAIN_inds_p7['ind5'],
                             TRAIN_inds_p8['ind5'], TRAIN_inds_p9['ind5'], TRAIN_inds_p10['ind5'],
                             TRAIN_inds_p11['ind5'], TRAIN_inds_p12['ind5'], TRAIN_inds_p13['ind5'],
                             TRAIN_inds_p14['ind5'], TRAIN_inds_p15['ind5'], TRAIN_inds_p16['ind5'],                                                                        TRAIN_inds_p17['ind5'], TRAIN_inds_p18['ind5'], TRAIN_inds_p19['ind5'],
                             TRAIN_inds_p20['ind5'],))

TRAIN_ind6 = np.concatenate((TRAIN_inds_p0['ind6'], TRAIN_inds_p1['ind6'], TRAIN_inds_p2['ind6'], TRAIN_inds_p3['ind6'],
                             TRAIN_inds_p4['ind6'], TRAIN_inds_p5['ind6'], TRAIN_inds_p6['ind6'], TRAIN_inds_p7['ind6'],
                             TRAIN_inds_p8['ind6'], TRAIN_inds_p9['ind6'], TRAIN_inds_p10['ind6'],
                             TRAIN_inds_p11['ind6'], TRAIN_inds_p12['ind6'], TRAIN_inds_p13['ind6'],
                             TRAIN_inds_p14['ind6'], TRAIN_inds_p15['ind6'], TRAIN_inds_p16['ind6'],                                                                        TRAIN_inds_p17['ind6'], TRAIN_inds_p18['ind6'], TRAIN_inds_p19['ind6'],
                             TRAIN_inds_p20['ind6'],))

TRAIN_ind7 = np.concatenate((TRAIN_inds_p0['ind7'], TRAIN_inds_p1['ind7'], TRAIN_inds_p2['ind7'], TRAIN_inds_p3['ind7'],
                             TRAIN_inds_p4['ind7'], TRAIN_inds_p5['ind7'], TRAIN_inds_p6['ind7'], TRAIN_inds_p7['ind7'],
                             TRAIN_inds_p8['ind7'], TRAIN_inds_p9['ind7'], TRAIN_inds_p10['ind7'],
                             TRAIN_inds_p11['ind7'], TRAIN_inds_p12['ind7'], TRAIN_inds_p13['ind7'],
                             TRAIN_inds_p14['ind7'], TRAIN_inds_p15['ind7'], TRAIN_inds_p16['ind7'],                                                                        TRAIN_inds_p17['ind7'], TRAIN_inds_p18['ind7'], TRAIN_inds_p19['ind7'],
                             TRAIN_inds_p20['ind7'],))

TRAIN_ind8 = np.concatenate((TRAIN_inds_p0['ind8'], TRAIN_inds_p1['ind8'], TRAIN_inds_p2['ind8'], TRAIN_inds_p3['ind8'],
                             TRAIN_inds_p4['ind8'], TRAIN_inds_p5['ind8'], TRAIN_inds_p6['ind8'], TRAIN_inds_p7['ind8'],
                             TRAIN_inds_p8['ind8'], TRAIN_inds_p9['ind8'], TRAIN_inds_p10['ind8'],
                             TRAIN_inds_p11['ind8'], TRAIN_inds_p12['ind8'], TRAIN_inds_p13['ind8'],
                             TRAIN_inds_p14['ind8'], TRAIN_inds_p15['ind8'], TRAIN_inds_p16['ind8'],                                                                        TRAIN_inds_p17['ind8'], TRAIN_inds_p18['ind8'], TRAIN_inds_p19['ind8'],
                             TRAIN_inds_p20['ind8'],))

TRAIN_ind9 = np.concatenate((TRAIN_inds_p0['ind9'], TRAIN_inds_p1['ind9'], TRAIN_inds_p2['ind9'], TRAIN_inds_p3['ind9'],
                             TRAIN_inds_p4['ind9'], TRAIN_inds_p5['ind9'], TRAIN_inds_p6['ind9'], TRAIN_inds_p7['ind9'],
                             TRAIN_inds_p8['ind9'], TRAIN_inds_p9['ind9'], TRAIN_inds_p10['ind9'],
                             TRAIN_inds_p11['ind9'], TRAIN_inds_p12['ind9'], TRAIN_inds_p13['ind9'],
                             TRAIN_inds_p14['ind9'], TRAIN_inds_p15['ind9'], TRAIN_inds_p16['ind9'],                                                                        TRAIN_inds_p17['ind9'], TRAIN_inds_p18['ind9'], TRAIN_inds_p19['ind9'],
                             TRAIN_inds_p20['ind9'],))

TRAIN_ind10 = np.concatenate((TRAIN_inds_p0['ind10'], TRAIN_inds_p1['ind10'], TRAIN_inds_p2['ind10'], TRAIN_inds_p3['ind10'],
                              TRAIN_inds_p4['ind10'], TRAIN_inds_p5['ind10'], TRAIN_inds_p6['ind10'], TRAIN_inds_p7['ind10'],
                              TRAIN_inds_p8['ind10'], TRAIN_inds_p9['ind10'], TRAIN_inds_p10['ind10'],
                              TRAIN_inds_p11['ind10'], TRAIN_inds_p12['ind10'], TRAIN_inds_p13['ind10'],
                              TRAIN_inds_p14['ind10'], TRAIN_inds_p15['ind10'], TRAIN_inds_p16['ind10'],                                                                     TRAIN_inds_p17['ind10'], TRAIN_inds_p18['ind10'], TRAIN_inds_p19['ind10'],
                              TRAIN_inds_p20['ind10'],))

TRAIN_ind11 = np.concatenate((TRAIN_inds_p0['ind11'], TRAIN_inds_p1['ind11'], TRAIN_inds_p2['ind11'], TRAIN_inds_p3['ind11'],
                              TRAIN_inds_p4['ind11'], TRAIN_inds_p5['ind11'], TRAIN_inds_p6['ind11'], TRAIN_inds_p7['ind11'],
                              TRAIN_inds_p8['ind11'], TRAIN_inds_p9['ind11'], TRAIN_inds_p10['ind11'],
                              TRAIN_inds_p11['ind11'], TRAIN_inds_p12['ind11'], TRAIN_inds_p13['ind11'],
                              TRAIN_inds_p14['ind11'], TRAIN_inds_p15['ind11'], TRAIN_inds_p16['ind11'],                                                                     TRAIN_inds_p17['ind11'], TRAIN_inds_p18['ind11'], TRAIN_inds_p19['ind11'],
                              TRAIN_inds_p20['ind11'],))

TRAIN_ind12 = np.concatenate((TRAIN_inds_p0['ind12'], TRAIN_inds_p1['ind12'], TRAIN_inds_p2['ind12'], TRAIN_inds_p3['ind12'],
                              TRAIN_inds_p4['ind12'], TRAIN_inds_p5['ind12'], TRAIN_inds_p6['ind12'], TRAIN_inds_p7['ind12'],
                              TRAIN_inds_p8['ind12'], TRAIN_inds_p9['ind12'], TRAIN_inds_p10['ind12'],
                              TRAIN_inds_p11['ind12'], TRAIN_inds_p12['ind12'], TRAIN_inds_p13['ind12'],
                              TRAIN_inds_p14['ind12'], TRAIN_inds_p15['ind12'], TRAIN_inds_p16['ind12'],                                                                     TRAIN_inds_p17['ind12'], TRAIN_inds_p18['ind12'], TRAIN_inds_p19['ind12'],
                              TRAIN_inds_p20['ind12'],))

TRAIN_ind13 = np.concatenate((TRAIN_inds_p0['ind13'], TRAIN_inds_p1['ind13'], TRAIN_inds_p2['ind13'], TRAIN_inds_p3['ind13'],
                              TRAIN_inds_p4['ind13'], TRAIN_inds_p5['ind13'], TRAIN_inds_p6['ind13'], TRAIN_inds_p7['ind13'],
                              TRAIN_inds_p8['ind13'], TRAIN_inds_p9['ind13'], TRAIN_inds_p10['ind13'],
                              TRAIN_inds_p11['ind13'], TRAIN_inds_p12['ind13'], TRAIN_inds_p13['ind13'],
                              TRAIN_inds_p14['ind13'], TRAIN_inds_p15['ind13'], TRAIN_inds_p16['ind13'],                                                                     TRAIN_inds_p17['ind13'], TRAIN_inds_p18['ind13'], TRAIN_inds_p19['ind13'],
                              TRAIN_inds_p20['ind13'],))

TRAIN_ind14 = np.concatenate((TRAIN_inds_p0['ind14'], TRAIN_inds_p1['ind14'], TRAIN_inds_p2['ind14'], TRAIN_inds_p3['ind14'],
                              TRAIN_inds_p4['ind14'], TRAIN_inds_p5['ind14'], TRAIN_inds_p6['ind14'], TRAIN_inds_p7['ind14'],
                              TRAIN_inds_p8['ind14'], TRAIN_inds_p9['ind14'], TRAIN_inds_p10['ind14'],
                              TRAIN_inds_p11['ind14'], TRAIN_inds_p12['ind14'], TRAIN_inds_p13['ind14'],
                              TRAIN_inds_p14['ind14'], TRAIN_inds_p15['ind14'], TRAIN_inds_p16['ind14'],                                                                     TRAIN_inds_p17['ind14'], TRAIN_inds_p18['ind14'], TRAIN_inds_p19['ind14'],
                              TRAIN_inds_p20['ind14'],))

TRAIN_ind15 = np.concatenate((TRAIN_inds_p0['ind15'], TRAIN_inds_p1['ind15'], TRAIN_inds_p2['ind15'], TRAIN_inds_p3['ind15'],
                              TRAIN_inds_p4['ind15'], TRAIN_inds_p5['ind15'], TRAIN_inds_p6['ind15'], TRAIN_inds_p7['ind15'],
                              TRAIN_inds_p8['ind15'], TRAIN_inds_p9['ind15'], TRAIN_inds_p10['ind15'],
                              TRAIN_inds_p11['ind15'], TRAIN_inds_p12['ind15'], TRAIN_inds_p13['ind15'],
                              TRAIN_inds_p14['ind15'], TRAIN_inds_p15['ind15'], TRAIN_inds_p16['ind15'],                                                                     TRAIN_inds_p17['ind15'], TRAIN_inds_p18['ind15'], TRAIN_inds_p19['ind15'],
                              TRAIN_inds_p20['ind15'],))

TRAIN_ind16 = np.concatenate((TRAIN_inds_p0['ind16'], TRAIN_inds_p1['ind16'], TRAIN_inds_p2['ind16'], TRAIN_inds_p3['ind16'],
                              TRAIN_inds_p4['ind16'], TRAIN_inds_p5['ind16'], TRAIN_inds_p6['ind16'], TRAIN_inds_p7['ind16'],
                              TRAIN_inds_p8['ind16'], TRAIN_inds_p9['ind16'], TRAIN_inds_p10['ind16'],
                              TRAIN_inds_p11['ind16'], TRAIN_inds_p12['ind16'], TRAIN_inds_p13['ind16'],
                              TRAIN_inds_p14['ind16'], TRAIN_inds_p15['ind16'], TRAIN_inds_p16['ind16'],                                                                     TRAIN_inds_p17['ind16'], TRAIN_inds_p18['ind16'], TRAIN_inds_p19['ind16'],
                              TRAIN_inds_p20['ind16'],))

TRAIN_ind17 = np.concatenate((TRAIN_inds_p0['ind17'], TRAIN_inds_p1['ind17'], TRAIN_inds_p2['ind17'], TRAIN_inds_p3['ind17'],
                              TRAIN_inds_p4['ind17'], TRAIN_inds_p5['ind17'], TRAIN_inds_p6['ind17'], TRAIN_inds_p7['ind17'],
                              TRAIN_inds_p8['ind17'], TRAIN_inds_p9['ind17'], TRAIN_inds_p10['ind17'],
                              TRAIN_inds_p11['ind17'], TRAIN_inds_p12['ind17'], TRAIN_inds_p13['ind17'],
                              TRAIN_inds_p14['ind17'], TRAIN_inds_p15['ind17'], TRAIN_inds_p16['ind17'],                                                                     TRAIN_inds_p17['ind17'], TRAIN_inds_p18['ind17'], TRAIN_inds_p19['ind17'],
                              TRAIN_inds_p20['ind17'],))

TRAIN_ind18 = np.concatenate((TRAIN_inds_p0['ind18'], TRAIN_inds_p1['ind18'], TRAIN_inds_p2['ind18'], TRAIN_inds_p3['ind18'],
                              TRAIN_inds_p4['ind18'], TRAIN_inds_p5['ind18'], TRAIN_inds_p6['ind18'], TRAIN_inds_p7['ind18'],
                              TRAIN_inds_p8['ind18'], TRAIN_inds_p9['ind18'], TRAIN_inds_p10['ind18'],
                              TRAIN_inds_p11['ind18'], TRAIN_inds_p12['ind18'], TRAIN_inds_p13['ind18'],
                              TRAIN_inds_p14['ind18'], TRAIN_inds_p15['ind18'], TRAIN_inds_p16['ind18'],                                                                     TRAIN_inds_p17['ind18'], TRAIN_inds_p18['ind18'], TRAIN_inds_p19['ind18'],
                              TRAIN_inds_p20['ind18'],))

TRAIN_ind19 = np.concatenate((TRAIN_inds_p0['ind19'], TRAIN_inds_p1['ind19'], TRAIN_inds_p2['ind19'], TRAIN_inds_p3['ind19'],
                              TRAIN_inds_p4['ind19'], TRAIN_inds_p5['ind19'], TRAIN_inds_p6['ind19'], TRAIN_inds_p7['ind19'],
                              TRAIN_inds_p8['ind19'], TRAIN_inds_p9['ind19'], TRAIN_inds_p10['ind19'],
                              TRAIN_inds_p11['ind19'], TRAIN_inds_p12['ind19'], TRAIN_inds_p13['ind19'],
                              TRAIN_inds_p14['ind19'], TRAIN_inds_p15['ind19'], TRAIN_inds_p16['ind19'],                                                                     TRAIN_inds_p17['ind19'], TRAIN_inds_p18['ind19'], TRAIN_inds_p19['ind19'],
                              TRAIN_inds_p20['ind19'],))

TRAIN_ind20 = np.concatenate((TRAIN_inds_p0['ind20'], TRAIN_inds_p1['ind20'], TRAIN_inds_p2['ind20'], TRAIN_inds_p3['ind20'],
                              TRAIN_inds_p4['ind20'], TRAIN_inds_p5['ind20'], TRAIN_inds_p6['ind20'], TRAIN_inds_p7['ind20'],
                              TRAIN_inds_p8['ind20'], TRAIN_inds_p9['ind20'], TRAIN_inds_p10['ind20'],
                              TRAIN_inds_p11['ind20'], TRAIN_inds_p12['ind20'], TRAIN_inds_p13['ind20'],
                              TRAIN_inds_p14['ind20'], TRAIN_inds_p15['ind20'], TRAIN_inds_p16['ind20'],                                                                     TRAIN_inds_p17['ind20'], TRAIN_inds_p18['ind20'], TRAIN_inds_p19['ind20'],
                              TRAIN_inds_p20['ind20'],))

TRAIN_ind21 = np.concatenate((TRAIN_inds_p0['ind21'], TRAIN_inds_p1['ind21'], TRAIN_inds_p2['ind21'], TRAIN_inds_p3['ind21'],
                              TRAIN_inds_p4['ind21'], TRAIN_inds_p5['ind21'], TRAIN_inds_p6['ind21'], TRAIN_inds_p7['ind21'],
                              TRAIN_inds_p8['ind21'], TRAIN_inds_p9['ind21'], TRAIN_inds_p10['ind21'],
                              TRAIN_inds_p11['ind21'], TRAIN_inds_p12['ind21'], TRAIN_inds_p13['ind21'],
                              TRAIN_inds_p14['ind21'], TRAIN_inds_p15['ind21'], TRAIN_inds_p16['ind21'],                                                                     TRAIN_inds_p17['ind21'], TRAIN_inds_p18['ind21'], TRAIN_inds_p19['ind21'],
                              TRAIN_inds_p20['ind21'],))

TRAIN_ind22 = np.concatenate((TRAIN_inds_p0['ind22'], TRAIN_inds_p1['ind22'], TRAIN_inds_p2['ind22'], TRAIN_inds_p3['ind22'],
                              TRAIN_inds_p4['ind22'], TRAIN_inds_p5['ind22'], TRAIN_inds_p6['ind22'], TRAIN_inds_p7['ind22'],
                              TRAIN_inds_p8['ind22'], TRAIN_inds_p9['ind22'], TRAIN_inds_p10['ind22'],
                              TRAIN_inds_p11['ind22'], TRAIN_inds_p12['ind22'], TRAIN_inds_p13['ind22'],
                              TRAIN_inds_p14['ind22'], TRAIN_inds_p15['ind22'], TRAIN_inds_p16['ind22'],                                                                     TRAIN_inds_p17['ind22'], TRAIN_inds_p18['ind22'], TRAIN_inds_p19['ind22'],
                              TRAIN_inds_p20['ind22'],))

TRAIN_ind23 = np.concatenate((TRAIN_inds_p0['ind23'], TRAIN_inds_p1['ind23'], TRAIN_inds_p2['ind23'], TRAIN_inds_p3['ind23'],
                              TRAIN_inds_p4['ind23'], TRAIN_inds_p5['ind23'], TRAIN_inds_p6['ind23'], TRAIN_inds_p7['ind23'],
                              TRAIN_inds_p8['ind23'], TRAIN_inds_p9['ind23'], TRAIN_inds_p10['ind23'],
                              TRAIN_inds_p11['ind23'], TRAIN_inds_p12['ind23'], TRAIN_inds_p13['ind23'],
                              TRAIN_inds_p14['ind23'], TRAIN_inds_p15['ind23'], TRAIN_inds_p16['ind23'],                                                                     TRAIN_inds_p17['ind23'], TRAIN_inds_p18['ind23'], TRAIN_inds_p19['ind23'],
                              TRAIN_inds_p20['ind23'],))


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

prefix = '/glade/work/ksha/NCAR/VALID_lead_inds_full_part{}.npy'

VALID_inds_p0 = np.load(prefix.format(0), allow_pickle=True)[()]
VALID_inds_p1 = np.load(prefix.format(1), allow_pickle=True)[()]
VALID_inds_p2 = np.load(prefix.format(2), allow_pickle=True)[()]
VALID_inds_p3 = np.load(prefix.format(3), allow_pickle=True)[()]
VALID_inds_p4 = np.load(prefix.format(4), allow_pickle=True)[()]
VALID_inds_p5 = np.load(prefix.format(5), allow_pickle=True)[()]
VALID_inds_p6 = np.load(prefix.format(6), allow_pickle=True)[()]
VALID_inds_p7 = np.load(prefix.format(6), allow_pickle=True)[()]
VALID_inds_p8 = np.load(prefix.format(6), allow_pickle=True)[()]


VALID_ind2 = np.concatenate((VALID_inds_p0['ind2'], VALID_inds_p1['ind2'], VALID_inds_p2['ind2'],
                             VALID_inds_p3['ind2'], VALID_inds_p4['ind2'], VALID_inds_p5['ind2'], 
                             VALID_inds_p6['ind2'], VALID_inds_p7['ind2'], VALID_inds_p8['ind2']))

VALID_ind3 = np.concatenate((VALID_inds_p0['ind3'], VALID_inds_p1['ind3'], VALID_inds_p2['ind3'],
                             VALID_inds_p3['ind3'], VALID_inds_p4['ind3'], VALID_inds_p5['ind3'], 
                             VALID_inds_p6['ind3'], VALID_inds_p7['ind3'], VALID_inds_p8['ind3']))

VALID_ind4 = np.concatenate((VALID_inds_p0['ind4'], VALID_inds_p1['ind4'], VALID_inds_p2['ind4'],
                             VALID_inds_p3['ind4'], VALID_inds_p4['ind4'], VALID_inds_p5['ind4'], 
                             VALID_inds_p6['ind4'], VALID_inds_p7['ind4'], VALID_inds_p8['ind4']))

VALID_ind5 = np.concatenate((VALID_inds_p0['ind5'], VALID_inds_p1['ind5'], VALID_inds_p2['ind5'],
                             VALID_inds_p3['ind5'], VALID_inds_p4['ind5'], VALID_inds_p5['ind5'], 
                             VALID_inds_p6['ind5'], VALID_inds_p7['ind5'], VALID_inds_p8['ind5']))

VALID_ind6 = np.concatenate((VALID_inds_p0['ind6'], VALID_inds_p1['ind6'], VALID_inds_p2['ind6'],
                             VALID_inds_p3['ind6'], VALID_inds_p4['ind6'], VALID_inds_p5['ind6'], 
                             VALID_inds_p6['ind6'], VALID_inds_p7['ind6'], VALID_inds_p8['ind6']))

VALID_ind7 = np.concatenate((VALID_inds_p0['ind7'], VALID_inds_p1['ind7'], VALID_inds_p2['ind7'],
                             VALID_inds_p3['ind7'], VALID_inds_p4['ind7'], VALID_inds_p5['ind7'], 
                             VALID_inds_p6['ind7'], VALID_inds_p7['ind7'], VALID_inds_p8['ind7']))

VALID_ind8 = np.concatenate((VALID_inds_p0['ind8'], VALID_inds_p1['ind8'], VALID_inds_p2['ind8'],
                             VALID_inds_p3['ind8'], VALID_inds_p4['ind8'], VALID_inds_p5['ind8'], 
                             VALID_inds_p6['ind8'], VALID_inds_p7['ind8'], VALID_inds_p8['ind8']))

VALID_ind9 = np.concatenate((VALID_inds_p0['ind9'], VALID_inds_p1['ind9'], VALID_inds_p2['ind9'],
                             VALID_inds_p3['ind9'], VALID_inds_p4['ind9'], VALID_inds_p5['ind9'], 
                             VALID_inds_p6['ind9'], VALID_inds_p7['ind9'], VALID_inds_p8['ind9']))

VALID_ind10 = np.concatenate((VALID_inds_p0['ind10'], VALID_inds_p1['ind10'], VALID_inds_p2['ind10'],
                              VALID_inds_p3['ind10'], VALID_inds_p4['ind10'], VALID_inds_p5['ind10'], 
                              VALID_inds_p6['ind10'], VALID_inds_p7['ind10'], VALID_inds_p8['ind10']))

VALID_ind11 = np.concatenate((VALID_inds_p0['ind11'], VALID_inds_p1['ind11'], VALID_inds_p2['ind11'],
                              VALID_inds_p3['ind11'], VALID_inds_p4['ind11'], VALID_inds_p5['ind11'], 
                              VALID_inds_p6['ind11'], VALID_inds_p7['ind11'], VALID_inds_p8['ind11']))

VALID_ind12 = np.concatenate((VALID_inds_p0['ind12'], VALID_inds_p1['ind12'], VALID_inds_p2['ind12'],
                              VALID_inds_p3['ind12'], VALID_inds_p4['ind12'], VALID_inds_p5['ind12'], 
                              VALID_inds_p6['ind12'], VALID_inds_p7['ind12'], VALID_inds_p8['ind12']))

VALID_ind13 = np.concatenate((VALID_inds_p0['ind13'], VALID_inds_p1['ind13'], VALID_inds_p2['ind13'],
                              VALID_inds_p3['ind13'], VALID_inds_p4['ind13'], VALID_inds_p5['ind13'], 
                              VALID_inds_p6['ind13'], VALID_inds_p7['ind13'], VALID_inds_p8['ind13']))

VALID_ind14 = np.concatenate((VALID_inds_p0['ind14'], VALID_inds_p1['ind14'], VALID_inds_p2['ind14'],
                              VALID_inds_p3['ind14'], VALID_inds_p4['ind14'], VALID_inds_p5['ind14'], 
                              VALID_inds_p6['ind14'], VALID_inds_p7['ind14'], VALID_inds_p8['ind14']))

VALID_ind15 = np.concatenate((VALID_inds_p0['ind15'], VALID_inds_p1['ind15'], VALID_inds_p2['ind15'],
                              VALID_inds_p3['ind15'], VALID_inds_p4['ind15'], VALID_inds_p5['ind15'], 
                              VALID_inds_p6['ind15'], VALID_inds_p7['ind15'], VALID_inds_p8['ind15']))

VALID_ind16 = np.concatenate((VALID_inds_p0['ind16'], VALID_inds_p1['ind16'], VALID_inds_p2['ind16'],
                              VALID_inds_p3['ind16'], VALID_inds_p4['ind16'], VALID_inds_p5['ind16'], 
                              VALID_inds_p6['ind16'], VALID_inds_p7['ind16'], VALID_inds_p8['ind16']))

VALID_ind17 = np.concatenate((VALID_inds_p0['ind17'], VALID_inds_p1['ind17'], VALID_inds_p2['ind17'],
                              VALID_inds_p3['ind17'], VALID_inds_p4['ind17'], VALID_inds_p5['ind17'], 
                              VALID_inds_p6['ind17'], VALID_inds_p7['ind17'], VALID_inds_p8['ind17']))

VALID_ind18 = np.concatenate((VALID_inds_p0['ind18'], VALID_inds_p1['ind18'], VALID_inds_p2['ind18'],
                              VALID_inds_p3['ind18'], VALID_inds_p4['ind18'], VALID_inds_p5['ind18'], 
                              VALID_inds_p6['ind18'], VALID_inds_p7['ind18'], VALID_inds_p8['ind18']))

VALID_ind19 = np.concatenate((VALID_inds_p0['ind19'], VALID_inds_p1['ind19'], VALID_inds_p2['ind19'],
                              VALID_inds_p3['ind19'], VALID_inds_p4['ind19'], VALID_inds_p5['ind19'], 
                              VALID_inds_p6['ind19'], VALID_inds_p7['ind19'], VALID_inds_p8['ind19']))

VALID_ind20 = np.concatenate((VALID_inds_p0['ind20'], VALID_inds_p1['ind20'], VALID_inds_p2['ind20'],
                              VALID_inds_p3['ind20'], VALID_inds_p4['ind20'], VALID_inds_p5['ind20'], 
                              VALID_inds_p6['ind20'], VALID_inds_p7['ind20'], VALID_inds_p8['ind20']))

VALID_ind21 = np.concatenate((VALID_inds_p0['ind21'], VALID_inds_p1['ind21'], VALID_inds_p2['ind21'],
                              VALID_inds_p3['ind21'], VALID_inds_p4['ind21'], VALID_inds_p5['ind21'], 
                              VALID_inds_p6['ind21'], VALID_inds_p7['ind21'], VALID_inds_p8['ind21']))

VALID_ind22 = np.concatenate((VALID_inds_p0['ind22'], VALID_inds_p1['ind22'], VALID_inds_p2['ind22'],
                              VALID_inds_p3['ind22'], VALID_inds_p4['ind22'], VALID_inds_p5['ind22'], 
                              VALID_inds_p6['ind22'], VALID_inds_p7['ind22'], VALID_inds_p8['ind22']))

VALID_ind23 = np.concatenate((VALID_inds_p0['ind23'], VALID_inds_p1['ind23'], VALID_inds_p2['ind23'],
                              VALID_inds_p3['ind23'], VALID_inds_p4['ind23'], VALID_inds_p5['ind23'], 
                              VALID_inds_p6['ind23'], VALID_inds_p7['ind23'], VALID_inds_p8['ind23']))

IND_VALID_lead = {}
IND_VALID_lead['lead2'] = VALID_ind2
IND_VALID_lead['lead3'] = VALID_ind3
IND_VALID_lead['lead4'] = VALID_ind4
IND_VALID_lead['lead5'] = VALID_ind5
IND_VALID_lead['lead6'] = VALID_ind6
IND_VALID_lead['lead7'] = VALID_ind7
IND_VALID_lead['lead8'] = VALID_ind8

IND_VALID_lead['lead9'] = VALID_ind9
IND_VALID_lead['lead10'] = VALID_ind10
IND_VALID_lead['lead11'] = VALID_ind11
IND_VALID_lead['lead12'] = VALID_ind12
IND_VALID_lead['lead13'] = VALID_ind13
IND_VALID_lead['lead14'] = VALID_ind14
IND_VALID_lead['lead15'] = VALID_ind15
IND_VALID_lead['lead16'] = VALID_ind16
IND_VALID_lead['lead17'] = VALID_ind17
IND_VALID_lead['lead18'] = VALID_ind18

IND_VALID_lead['lead19'] = VALID_ind19
IND_VALID_lead['lead20'] = VALID_ind20
IND_VALID_lead['lead21'] = VALID_ind21
IND_VALID_lead['lead22'] = VALID_ind22
IND_VALID_lead['lead23'] = VALID_ind23

np.save('/glade/work/ksha/NCAR/IND_TRAIN_lead_full.npy', IND_TRAIN_lead)
print('/glade/work/ksha/NCAR/IND_TRAIN_lead_full.npy')

np.save('/glade/work/ksha/NCAR/IND_VALID_lead_full.npy', IND_VALID_lead)
print('/glade/work/ksha/NCAR/IND_VALID_lead_full.npy')


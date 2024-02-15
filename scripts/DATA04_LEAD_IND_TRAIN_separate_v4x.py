
import numpy as np

prefix = '/glade/work/ksha/NCAR/TRAIN_lead_inds_{}{}{}{}_part{}_v4x.npy'

N_parts = 4

for lead1_base in range(2, 21):
    
    TRAIN_ind_lead1 = np.array([])
    TRAIN_ind_lead2 = np.array([])
    TRAIN_ind_lead3 = np.array([])
    TRAIN_ind_lead4 = np.array([])
    
    count = 0
    
    lead2_base = lead1_base + 1
    lead3_base = lead1_base + 2
    lead4_base = lead1_base + 3
    
    for parts in range(N_parts):

        TRAIN_inds  = np.load(prefix.format(lead1_base, lead2_base, lead3_base, lead4_base, parts), allow_pickle=True)[()]
    
        ind_lead1 = TRAIN_inds['ind{}'.format(lead1_base)]
        ind_lead2 = TRAIN_inds['ind{}'.format(lead2_base)]
        ind_lead3 = TRAIN_inds['ind{}'.format(lead3_base)]
        ind_lead4 = TRAIN_inds['ind{}'.format(lead4_base)]

        TRAIN_ind_lead1 = np.concatenate((TRAIN_ind_lead1, ind_lead1), axis=0)
        TRAIN_ind_lead2 = np.concatenate((TRAIN_ind_lead2, ind_lead2), axis=0)
        TRAIN_ind_lead3 = np.concatenate((TRAIN_ind_lead3, ind_lead3), axis=0)
        TRAIN_ind_lead4 = np.concatenate((TRAIN_ind_lead4, ind_lead4), axis=0)

    IND_TRAIN_lead = {}
    IND_TRAIN_lead['lead{}'.format(lead1_base)] = TRAIN_ind_lead1
    IND_TRAIN_lead['lead{}'.format(lead2_base)] = TRAIN_ind_lead2
    IND_TRAIN_lead['lead{}'.format(lead3_base)] = TRAIN_ind_lead3
    IND_TRAIN_lead['lead{}'.format(lead4_base)] = TRAIN_ind_lead4
    
    np.save('/glade/work/ksha/NCAR/IND_TRAIN_lead_{}{}{}{}_v4x.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base), IND_TRAIN_lead)
    print('/glade/work/ksha/NCAR/IND_TRAIN_lead_{}{}{}{}_v4x.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base))





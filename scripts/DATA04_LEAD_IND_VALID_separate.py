
import numpy as np

prefix = '/glade/work/ksha/NCAR/VALID_lead_inds_{}{}{}{}_part{}.npy'

N_parts = 8

for lead1_base in range(2, 21):
    
    VALID_ind_lead1 = np.array([])
    VALID_ind_lead2 = np.array([])
    VALID_ind_lead3 = np.array([])
    VALID_ind_lead4 = np.array([])
    
    count = 0
    
    lead2_base = lead1_base + 1
    lead3_base = lead1_base + 2
    lead4_base = lead1_base + 3
    
    for parts in range(N_parts):

        VALID_inds  = np.load(prefix.format(lead1_base, lead2_base, lead3_base, lead4_base, parts), allow_pickle=True)[()]
    
        ind_lead1 = VALID_inds['ind{}'.format(lead1_base)]
        ind_lead2 = VALID_inds['ind{}'.format(lead2_base)]
        ind_lead3 = VALID_inds['ind{}'.format(lead3_base)]
        ind_lead4 = VALID_inds['ind{}'.format(lead4_base)]

        VALID_ind_lead1 = np.concatenate((VALID_ind_lead1, ind_lead1), axis=0)
        VALID_ind_lead2 = np.concatenate((VALID_ind_lead2, ind_lead2), axis=0)
        VALID_ind_lead3 = np.concatenate((VALID_ind_lead3, ind_lead3), axis=0)
        VALID_ind_lead4 = np.concatenate((VALID_ind_lead4, ind_lead4), axis=0)

    IND_VALID_lead = {}
    IND_VALID_lead['lead{}'.format(lead1_base)] = VALID_ind_lead1
    IND_VALID_lead['lead{}'.format(lead2_base)] = VALID_ind_lead2
    IND_VALID_lead['lead{}'.format(lead3_base)] = VALID_ind_lead3
    IND_VALID_lead['lead{}'.format(lead4_base)] = VALID_ind_lead4
    
    np.save('/glade/work/ksha/NCAR/IND_VALID_lead_{}{}{}{}.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base), IND_VALID_lead)
    print('/glade/work/ksha/NCAR/IND_VALID_lead_{}{}{}{}.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base))





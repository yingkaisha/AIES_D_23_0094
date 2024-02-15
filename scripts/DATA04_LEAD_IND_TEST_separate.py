
import numpy as np

prefix = '/glade/work/ksha/NCAR/TEST_lead_inds_{}{}{}{}_part{}.npy'

N_parts = 12

for lead1_base in range(2, 21):
    
    TEST_ind_lead1 = np.array([])
    TEST_ind_lead2 = np.array([])
    TEST_ind_lead3 = np.array([])
    TEST_ind_lead4 = np.array([])
    
    count = 0
    
    lead2_base = lead1_base + 1
    lead3_base = lead1_base + 2
    lead4_base = lead1_base + 3
    
    for parts in range(N_parts):

        TEST_inds  = np.load(prefix.format(lead1_base, lead2_base, lead3_base, lead4_base, parts), allow_pickle=True)[()]
    
        ind_lead1 = TEST_inds['ind{}'.format(lead1_base)]
        ind_lead2 = TEST_inds['ind{}'.format(lead2_base)]
        ind_lead3 = TEST_inds['ind{}'.format(lead3_base)]
        ind_lead4 = TEST_inds['ind{}'.format(lead4_base)]

        TEST_ind_lead1 = np.concatenate((TEST_ind_lead1, ind_lead1), axis=0)
        TEST_ind_lead2 = np.concatenate((TEST_ind_lead2, ind_lead2), axis=0)
        TEST_ind_lead3 = np.concatenate((TEST_ind_lead3, ind_lead3), axis=0)
        TEST_ind_lead4 = np.concatenate((TEST_ind_lead4, ind_lead4), axis=0)

    IND_TEST_lead = {}
    IND_TEST_lead['lead{}'.format(lead1_base)] = TEST_ind_lead1
    IND_TEST_lead['lead{}'.format(lead2_base)] = TEST_ind_lead2
    IND_TEST_lead['lead{}'.format(lead3_base)] = TEST_ind_lead3
    IND_TEST_lead['lead{}'.format(lead4_base)] = TEST_ind_lead4
    
    np.save('/glade/work/ksha/NCAR/IND_TEST_lead_{}{}{}{}.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base), IND_TEST_lead)
    print('/glade/work/ksha/NCAR/IND_TEST_lead_{}{}{}{}.npy'.format(lead1_base, lead2_base, lead3_base, lead4_base))





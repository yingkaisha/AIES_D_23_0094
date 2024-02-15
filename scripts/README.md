# The scripts folder

* Run `DATA00_HRRRv3_subset.py` and `DATA00_HRRRv4_subset.py` to subset predictors from HRRR .nat files.
* Run `DATA01_HRRRv3_predictor_stats_patch.py` and `DATA01_HRRRv4x_predictor_stats_patch.py` to get the mean and std for each 64-by-64 patch (the scripts produce .npy files for each patch inidividually).
* Run `DATA02_HRRRv3_patch_stats_combine.py` and `DATA02_HRRRv4_patch_stats_combine.py` to merge inidividual .npy files
* Run `DATA03_BATCH_gen_v3.py` and `DATA03_BATCH_gen_v4.py` to generate training and validation batches.
* Run `CNN01_base_vgg.py` to train the representation learning model.
* Run `CNN01_GAN_CREF.py` and `CNN01_GAN_ENVI.py` to train the two CGANs.
* Run `CNN00_feature_vector_GAN_testing.py` to let CGANs produce ensemble members.
* Run `DATA04_LEAD_IND_TRAIN_separate.py`, `DATA04_LEAD_IND_VALID_separate.py`, and `DATA04_LEAD_IND_TEST_separate.py` to check the consistency of the training/validation/testing sets and identify the 4-hour forecast lead time windows.
* Run `CNN00_feature_vector_basic_training.py`, `CNN00_feature_vector_basic_validation.py`, and `CNN00_feature_vector_basic_testing.py` to produce feature vectors from the representation learning model.
* Run `CNN02_classifier_train_cgan_lead2.py` , `CNN02_classifier_train_cgan_lead3.py` , and `CNN02_classifier_train_cgan_lead4.py` to train the severe weather prediction model
* Run `CNN03_classifier_inf_cgan_lead2.py`, `CNN03_classifier_inf_cgan_lead3.py`, and `CNN03_classifier_inf_cgan_lead4.py` to predict severe weather probabilities.


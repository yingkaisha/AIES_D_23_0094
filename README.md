# Generative ensemble deep learning severe weather prediction from a deterministic convection-allowing model

Yingkai Sha, Ryan A. Sobash, David John Gagne II

National Center for Atmospheric Research, Boulder, Colorado, USA

## Abstract

An ensemble post-processing method is developed for the probabilistic prediction of severe weather (tornadoes, hail, and wind gusts) over the conterminous United States (CONUS). The method combines conditional generative adversarial networks (CGANs), a type of deep generative model, with a convolutional neural network (CNN) to post-process convection-allowing model (CAM) forecasts. The CGANs are designed to create synthetic ensemble members from deterministic CAM forecasts, and their outputs are processed by the CNN to estimate the probability of severe weather. The method is tested using High-Resolution Rapid Refresh (HRRR) 1--24 hr forecasts as inputs and Storm Prediction Center (SPC) severe weather reports as targets. The method produced skillful predictions with up to 20\% Brier Skill Score (BSS) increases compared to other neural-network-based reference methods using a testing dataset of HRRR forecasts in 2021. For the evaluation of uncertainty quantification, the method is overconfident but produces meaningful ensemble spreads that can distinguish good and bad forecasts. The quality of CGAN outputs is also evaluated. Results show that the CGAN outputs behave similarly to a numerical ensemble; they preserved the inter-variable correlations and the contribution of influential predictors as in the original HRRR forecasts. This work provides a novel approach to post-process CAM output using neural networks that can be applied to severe weather prediction.

Sha, Y., R. A. Sobash, D. J. Gagne II, 2023: Generative ensemble deep learning severe weather prediction from a deterministic convection-allowing model. in review: Artificial Intelligence for the Earth Systems. pre-print: http://arxiv.org/abs/2310.06045

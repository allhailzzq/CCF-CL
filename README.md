# CCF-CL
Sample code for paper Forecasting the Clinical Status of a Patient Through Contrastive Learning.

The model training has two steps: 1) contrastive pretraining with contrastive_training.py; 2) transfer learning (for code prediction task) with transfer_naive.py. Specifically, the class FeatureNet is for representation learning, while the class PredNet is for the prediction task.

This sample code uses a simple LSTM backbone with three inputs: code.npy (diagnosis and procedure codes index as numbers, e.g., 0, 1, ...); length.npy (the number of visits for each patient); aux_info.npy (auxiliary information, e.g., the demographic, time interval between visits, etc.,one-hot encoded). Please construct your own input according to your data and tasks. Also you can use other more up-to-date SOTA backbones like transformers to model the patient visit sequence, and change the hyper-parameters for data augmentation that best fit your data.

Please refer to the paper for the logic behind this code. The code is based on tensorflow 2.4, which may be outdated as it was written more than three years ago. So please use it only as a sample to learn about the implementation of the algorithm. 

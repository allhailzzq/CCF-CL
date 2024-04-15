# CCF-CL
Sample code for paper Forecasting the Clinical Status of a Patient Through Contrastive Learning.

The model training has two steps: 1) contrastive pretraining with contrastive_training.py; 2) transfer learning (for code prediction task) with transfer_naive.py.

This sample code uses a simple LSTM backbone with three inputs: code.npy (diagnosis and procedure codes); length.npy (the number of visits for each patient); aux_info.npy (auxiliary information, e.g., the demographic, time interval between visits, etc.). Please construct your own input according to your data and tasks.

Please refer to the paper for the logic behind this code.

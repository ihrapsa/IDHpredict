Use those scripts* if you want to split your pipeline into 2 parts: 
1. Script `main.py` does preproccessing and tumor segmentation calling MODEL1
2. Script `norm_pred.py` does the normalization of features and the prediction of `IDH` status by calling MODEL2

*_You must copy them into parent directory._

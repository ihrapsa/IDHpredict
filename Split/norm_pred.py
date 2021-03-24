import os
import dill
import pandas as pd
import os.path
from os import path

#create dataframe with prediction values
if path.exists('OUTPUT/predict.csv') == True:
	predict = pd.read_csv('OUTPUT/predict.csv')
else:
	predict = pd.DataFrame (columns = ['id','IDH'])   


arr = os.listdir("INPUT")

def norm_pred(x, t1c_resnet_arr, t2_resnet_arr, predmask_arr):
	##### Normalize features
	sla_features = pd.read_csv('OUTPUT/'+x+'/sla_features.csv')
	sla_features_norm = np.array(sla_features)-np.array(sla_features_mean)

	sla_arr = np.repeat(sla_features_norm, 5, axis=0)

	##### Get probability of IDH mutation from Model 2
	output_mean = get_IDH_pred(t1c_resnet_arr, t2_resnet_arr, predmask_arr, sla_arr)

	return output_mean.cpu().item()

for caz in arr:
	print(caz)
	dill.load_session('OUTPUT/'+caz+'/session.db')

	IDH = norm_pred(caz, t1c_resnet_arr, t2_resnet_arr, predmask_arr)
	#populate predict dataframe with predicted IDH mutation
	predict = predict.append({
         'id': caz ,
         'IDH': IDH
          }, ignore_index=True)
	predict.to_csv('OUTPUT/predict.csv', index=False)


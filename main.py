import os
import pandas as pd
from img_processing import *
from datetime import datetime

started = datetime.now()

current_time = started.strftime("%H:%M:%S")
print("Time run started =", current_time)


def model_testing(pathin, pathout):
    # ### Image processing - skullstripping, resampling, registration and bias correction

    # file path of original T1C 
    T1C_original_file = pathin+'T1C.nii.gz'            
    T2_original_file = pathin+'T2.nii.gz'
    FLAIR_original_file = pathin+'FLAIR.nii.gz'

    # file path of skull-stripped images
    T1C_bet_file = pathout+'t1c_bet.nii.gz'            
    T2_bet_file = pathout+'t2_bet.nii.gz'
    FLAIR_bet_file = pathout+'flair_bet.nii.gz'

    # file path of the mask for T1C skull stripping
    brainmask_T1C_file = pathout+'T1C_bet0_temp_mask.nii.gz'  

    # filenames to save isovoxel images / brain mask
    T1C_iso_file = pathout+'t1c_isovoxel.nii.gz'
    T2_iso_file = pathout+'t2_isovoxel.nii.gz'
    FLAIR_iso_file = pathout+'flair_isovoxel.nii.gz'
    brainmask_iso_file = pathout+'mask_brain_isovoxel.nii.gz'    

    # filenames to save bias-corrected images   
    T1C_corrected_file = pathout+'t1c_corrected.nii.gz'
    T2_corrected_file = pathout+'t2_corrected.nii.gz'
    FLAIR_corrected_file = pathout+'flair_corrected.nii.gz'

    ## filename for preliminary skull-stripped T1C                                       
    T1C_bet_temp_file = pathout+'T1C_bet0_temp.nii.gz'

    func_img_proc(T1C_original_file, T2_original_file, FLAIR_original_file, T1C_bet_file, T2_bet_file, FLAIR_bet_file, brainmask_T1C_file, T1C_iso_file, T2_iso_file, FLAIR_iso_file, brainmask_iso_file, T1C_corrected_file, T2_corrected_file, FLAIR_corrected_file, T1C_bet_temp_file)


    ################################## Model 1 : Automatic tumor segmentation  ############################################

    #### Preprocessing for Model 1
    (t1c_unet_arr, flair_unet_arr, cropdown_info) = func_norm_model1(T1C_corrected_file, FLAIR_corrected_file, brainmask_iso_file)
    # cropdown_info will be used for resmampling the predicted tumor mask to original isovoxel space, and preprcessing for Model2.
    ##### Get tumor mask from Model 1

    predmask_arr = func_get_predmask(t1c_unet_arr, flair_unet_arr)

    # #### Resample the predicted mask back to original isovoxel space

    predmask_isovoxel_arr = func_mask_back2iso(predmask_arr, cropdown_info)
    predmask_isovoxel_arr_sitk = np.transpose(predmask_isovoxel_arr, (2,1,0))
    predmask_isovoxel_img = sitk.GetImageFromArray(predmask_isovoxel_arr_sitk)

    predmask_isovoxel_file = 'predmask_isovoxel.nii.gz' #filename for predicted mask of isovoxel resolution
    sitk.WriteImage(predmask_isovoxel_img, predmask_isovoxel_file)   # save the automatic segmentation of isovoxel resolution


    ################################## Model 2 : CNN classifier for IDH status prediction ###################################

    # #### Preprocessing for Model 2
    t1c_corrected_img = nb.load(T1C_corrected_file)
    t1c_corrected_arr = t1c_corrected_img.get_data()
    t2_corrected_img = nb.load(T2_corrected_file)
    t2_corrected_arr = t2_corrected_img.get_data()
    brain_mask = nb.load(brainmask_iso_file)
    brain_mask_arr = brain_mask.get_data()

    t1c_resnet_arr = func_norm_resnet(t1c_corrected_arr, predmask_isovoxel_arr, brain_mask_arr, cropdown_info)
    t2_resnet_arr = func_norm_resnet(t2_corrected_arr, predmask_isovoxel_arr, brain_mask_arr, cropdown_info)

    # #### Get shape and loci features from tumor mask of 1mm isovoxel

    sla_features = func_shapeloci(T1C_iso_file, predmask_isovoxel_file)

    # #### Add patient's age

    age = age_df.loc[caz]['age']
    sla_features['age'] = pd.Series(age)  

    # #### Normalize features

    sla_features_norm = sla_features

    for i in range(len(sla_features.columns)):
        sla_features_norm.iloc[0, i] = (sla_features.iloc[0, i] - sla_features_mean.iloc[0, i])/sla_features_std.iloc[0, i]

    # ####

    sla_features_norm = np.array(sla_features)
    sla_arr = np.repeat(sla_features_norm, 5, axis=0)
    sla_arr.shape

    # #### Get probability of IDH mutation from Model 2

    output_mean = get_IDH_pred(t1c_resnet_arr, t2_resnet_arr, predmask_arr, sla_arr)

    return output_mean.cpu().item()


    
##########################################################
###### Pipeline ##########################################


#read csv files for normalization
sla_features_mean = pd.read_csv('sla_features_mean.csv')
sla_features_std = pd.read_csv('sla_features_std.csv')
#read csv file with age data
age_df = pd.read_csv('age.csv')
age_df = age_df.set_index('id')
#create dataframe with prediction values
predict = pd.DataFrame (columns = ['id','IDH'])   


arr = os.listdir("INPUT")
for caz in arr:
    #convert DICOM to NIFTI
    os.mkdir("NIFTI/"+caz)
    os.system('./dcm2niix -f "%f" -p y -z y -b n -o "NIFTI/'+caz+'/" "INPUT/'+caz+'/FLAIR"')
    os.system('./dcm2niix -f "%f" -p y -z y -b n -o "NIFTI/'+caz+'/" "INPUT/'+caz+'/T1C"')
    os.system('./dcm2niix -f "%f" -p y -z y -b n -o "NIFTI/'+caz+'/" "INPUT/'+caz+'/T2"')
    #Image proccessing, resampling, correction
    os.mkdir("OUTPUT/"+caz)

    IDH = model_testing("./NIFTI/"+caz+"/","./OUTPUT/"+caz+"/")
    
    #populate predict dataframe with predicted IDH mutation
    predict = predict.append({
         'id': caz ,
         'IDH': IDH
          }, ignore_index=True)

    #export predict dataframe to csv 
    print("Exporting results to 'predict.csv'")
    predict.to_csv('OUTPUT/predict.csv', index=False)

    print ("Prediction of case", caz, "finished")
    #Print the amount of time taken/case
    now = datetime.now()
    took = now - started
    print(caz, "Prediction took ", took)


ended = datetime.now()
took = ended - started

print("Run took=", took)

###############################################################################
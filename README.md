# IDHpredict
This is a public repository for ["Fully Automated Hybrid Network to Predict IDH Mutation Status of Glioma via Deep Learning and Radiomics"](https://academic.oup.com/neuro-oncology/article-abstract/23/2/304/5876011) by Choi et al. The core code is entirely based on [yoonchoi-neuro/automated_hybrid_IDH](https://github.com/yoonchoi-neuro/automated_hybrid_IDH).

 The automated hybrid model consists of UNet-based Model1 for tumor segmentation, ResNet-based Model 2 for IDH status prediction, and automated processing pipeline inbetween. Model 2 integrates 2D MR images, radiomic features of 3D tumor shape & loci, and age in one CNN. 

--------------------------------------------------------------------
![alt_text](https://github.com/ihrapsa/automated_hybrid_IDH/blob/master/workflow.png)
--------------------------------------------------------------------
I've modified and adapted the code into a script in order to test multiple patients on a single run. I've added the conversion part from DICOM `.dcm` to NIFTI `nii.gz` using the `dcm2niix` tool by [rordenlab](https://github.com/rordenlab/dcm2niix).  I've also added the skulstripping part that is missing from the original repo. 


 * Python module requirements : Nipype / FSL / ANTs / PyRadiomics / PyTorch
 * The process resquires GPU.


### How to use the script:

1. To test your cases, create inside `./INPUT` a separate directory for each patient and rename it to its unique `id`. Inside each patient's directory put the 3 axial MRI DICOM directories renamed to: `T1C`, `T2` and `FLAIR`.
2. Edit the `age.csv` file and populate it with your patients' `age` and `id`.
3. Run `main.py`
4. The script outputs a `predict.csv` file inside `./OUTPUT` where all prediction scores are listed alongside patient's `id`.

* Inside `Split` directory the script was split into two to separate image proccesing and MODEL1 from MODEL2. 

* The code to test one sample using Jupyter-Notebook is avaialble at https://github.com/ihrapsa/automated_hybrid_IDH

_____________________________________________________

#### Credits:
[yoonchoi-neuro](https://github.com/yoonchoi-neuro) - for the hard work she put into this

_____________________________________________________

#### Contact:

💬 discord: jonah1024#4422  
:email: email: hrapsaiona@gmail.com

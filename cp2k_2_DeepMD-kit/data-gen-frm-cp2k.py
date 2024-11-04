-import dpdata
from cp2kdata import Cp2kOutput
import numpy as np

cp2kmd_dir = input("Please, enter the output folder name: ")
cp2kmd_output_name = input("Please, enter the log file name: ")
dp = dpdata.LabeledSystem(cp2kmd_dir, cp2k_output_name=cp2kmd_output_name, fmt="cp2kdata/md")
print(dp)

n = int(input("Please, enter the number of frames for training: "))
# random choose for training_data
index_training = np.random.choice(len(dp), size=n, replace=False)
# other indexes are validation_data
index_validation = list(set(range(len(dp)))-set(index_training))       
data_training = dp.sub_system(index_training)
data_validation = dp.sub_system(index_validation)
# all training data put into directory:"training_data" 
data_training.to_deepmd_npy('training_data')               
# all validation data put into directory:"validation_data"
data_validation.to_deepmd_npy('validation_data')           
print('# the training data contains %d frames' % len(data_training)) 
print('# the validation data contains %d frames' % len(data_validation)) 

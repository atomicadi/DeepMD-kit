import dpdata 
import numpy as np
data = dpdata.LabeledSystem('OUTCAR', fmt = 'vasp/outcar') 
print('# the data contains %d frames' % len(data))
n = int(input("Please, write the number of frames for training: "))
# random choose for training_data
index_training = np.random.choice(len(data), size=n, replace=False)
# other indexes are validation_data
index_validation = list(set(range(len(data)))-set(index_training))       
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)
# all training data put into directory:"training_data" 
data_training.to_deepmd_npy('training_data')               
# all validation data put into directory:"validation_data"
data_validation.to_deepmd_npy('validation_data')           
print('# the training data contains %d frames' % len(data_training)) 
print('# the validation data contains %d frames' % len(data_validation)) 
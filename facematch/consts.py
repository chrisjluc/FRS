import os

# Model
NN1 = 'nn1'
NN2 = 'nn2'

# Image processing
norm_shape = 230, 230
nn1_input_shape = 100, 100
nn2_input_shape = 165, 120
noise_width = 15

# Storage
data_path = '.data'
image_path = os.path.join(data_path, 'images')
model_path = os.path.join(data_path, 'models')
image_ext = '_image.npy'
landmarks_ext = '_landmarks.npy'
features_ext = '_features.npy'

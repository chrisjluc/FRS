import os

# Model
NN1 = 'nn1'
NN2 = 'nn2'

cnn_batch_size = 32
cnn_validation_split = .15
cnn_nb_epoch = 10
cnn_activation_batch_size = 512

sae_batch_size = 256
sae_nb_epoch = 10
sae_validation_split =.15
sae_p1_input_size = 3584
sae_p1_encoding_size = 2048
sae_p2_encoding_size = 1024
sae_p3_encoding_size = 512

# Image processing
norm_shape = 230, 230
nn1_input_shape = 1, 100, 100
nn2_input_shape = 1, 165, 120
noise_width = 15

# Storage
data_path = '/home/ubuntu/FRS/facematch/.data'
face_predictor_path = os.path.join(
        data_path, 'shape_predictor_68_face_landmarks.dat')
image_path = os.path.join(data_path, 'images')
model_path = os.path.join(data_path, 'models')
image_ext = '_image.npy'
landmarks_ext = '_landmarks.npy'
features_ext = '_features.npy'
activations_ext = '_activations.npy'
encoder_ext = '_sae_encoder.npy'
autoencoder_ext = '_sae_autoencoder.npy'
json_ext = '.json'
h5_ext = '.h5'

# Tasks
num_gpus = 4

import consts

import image_processing as ip
import data_processing as dp
import numpy as np

from image import Image
from storage import Reader, Writer
from models import NN1Model, NN2Model, AutoEncoderModel
from tasks import TrainingTask, ActivationExtractionTask, TrainingAutoEncoderTask
from task_manager import TaskManager

class API(object):

    def __init__(self):
        self.reader = Reader()
        self.writer = Writer()

    def load_model(self):
        """
        Retrieves model if it has been trained
        """
        user_ids = self.reader.get_user_ids()
        self.models = [
                NN2Model(consts.cnn_h1, user_ids),
                NN1Model(consts.cnn_p1, user_ids),
                NN1Model(consts.cnn_p2, user_ids),
                NN1Model(consts.cnn_p3, user_ids),
                NN1Model(consts.cnn_p4, user_ids),
                NN1Model(consts.cnn_p5, user_ids),
                NN1Model(consts.cnn_p6, user_ids),
                AutoEncoderModel(consts.sae_p1),
                AutoEncoderModel(consts.sae_p2),
                AutoEncoderModel(consts.sae_p3),
                ]

        for model in self.models:
            model.load()

    def get_face_vector(self, image):
        im = Image(image)
        data = dp.create_training_data_for_mmdfr([[im]])
        cnn_models = self.models[:7]
        sae_models = self.models[7:]
        activations = np.concatenate(
                tuple(model.get_activations(_data) for model, _data in zip(cnn_models, data)),
                axis=1)
        for model in sae_models:
            activations = model.get_activations(activations)
        return activations

    def compute_score(self, face_vec_1, face_vec_2):
        pass

    def train(self):
        """
        Trains the model on all images that are currently in storage
        """
        #TODO:  Allow user to set model name
        cnn_h1 = consts.cnn_h1
        cnn_p1 = consts.cnn_p1
        cnn_p2 = consts.cnn_p2
        cnn_p3 = consts.cnn_p3
        cnn_p4 = consts.cnn_p4
        cnn_p5 = consts.cnn_p5
        cnn_p6 = consts.cnn_p6
        sae_p1 = consts.sae_p1
        sae_p2 = consts.sae_p2
        sae_p3 = consts.sae_p3

        user_ids = self.reader.get_user_ids()
        images = []
        for user_id in user_ids:
            images.append(self.reader.get_images(user_id))
        # Data augmentation
        cloned_images = dp.clone(images, 1)
        reflected_images = ip.apply_reflection(cloned_images)
        # Combine reflected and normal images
        images = dp.merge(images, reflected_images)
        images = dp.clone(images, 2)
        # Apply Random gaussian noise
        images = ip.apply_noise(images)
        data = dp.create_training_data_for_mmdfr(images)
        data_h1, data_p1, data_p2, data_p3, data_p4, data_p5, data_p6, data_y = data

        # Training CNNs
        tasks = [
            TrainingTask(NN2Model, data_h1, data_y, user_ids, cnn_h1),
            TrainingTask(NN1Model, data_p1, data_y, user_ids, cnn_p1),
            TrainingTask(NN1Model, data_p2, data_y, user_ids, cnn_p2),
            TrainingTask(NN1Model, data_p3, data_y, user_ids, cnn_p3),
            TrainingTask(NN1Model, data_p4, data_y, user_ids, cnn_p4),
            TrainingTask(NN1Model, data_p5, data_y, user_ids, cnn_p5),
            TrainingTask(NN1Model, data_p6, data_y, user_ids, cnn_p6)
            ]
        task_manager = TaskManager(tasks)
        task_manager.run_tasks()

        # Extracting activations from 2nd last layer for SAE
        tasks = [
            ActivationExtractionTask(NN2Model, cnn_h1, data_h1, user_ids),
            ActivationExtractionTask(NN1Model, cnn_p1, data_p1, user_ids),
            ActivationExtractionTask(NN1Model, cnn_p2, data_p2, user_ids),
            ActivationExtractionTask(NN1Model, cnn_p3, data_p3, user_ids),
            ActivationExtractionTask(NN1Model, cnn_p4, data_p4, user_ids),
            ActivationExtractionTask(NN1Model, cnn_p5, data_p5, user_ids),
            ActivationExtractionTask(NN1Model, cnn_p6, data_p6, user_ids)
            ]
        task_manager = TaskManager(tasks)
        task_manager.run_tasks()

        # Training first layer in SAE
        activations = np.concatenate((
            self.reader.load_activations(cnn_h1),
            self.reader.load_activations(cnn_p1),
            self.reader.load_activations(cnn_p2),
            self.reader.load_activations(cnn_p3),
            self.reader.load_activations(cnn_p4),
            self.reader.load_activations(cnn_p5),
            self.reader.load_activations(cnn_p6)
            ), axis=1)
        tasks = [TrainingAutoEncoderTask(sae_p1, activations, consts.sae_p1_input_size, consts.sae_p1_encoding_size)]
        task_manager = TaskManager(tasks)
        task_manager.run_tasks()

        # Training second layer in SAE
        activations = self.reader.load_activations(sae_p1)
        tasks = [TrainingAutoEncoderTask(sae_p2, activations, consts.sae_p1_encoding_size, consts.sae_p2_encoding_size)]
        task_manager = TaskManager(tasks)
        task_manager.run_tasks()

        # Training third layer in SAE
        activations = self.reader.load_activations(sae_p2)
        tasks = [TrainingAutoEncoderTask(sae_p3, activations, consts.sae_p2_encoding_size, consts.sae_p3_encoding_size)]
        task_manager = TaskManager(tasks)
        task_manager.run_tasks()

    def add_image(self, user_id, image):
        """
        params:
        user_id: string or int
        image: either a string of the filename or numpy array of an RGB image
        """
        im = Image(image)
        self.writer.save_image(user_id, im)

    def remove_images(self):
        """
        Removes all images that have been persisted
        in consts.image_path
        """
        self.writer.remove_directory(consts.image_path)
        self.writer.create_directory(consts.image_path)

import consts

import image_processing as ip
import data_processing as dp

from image import Image
from storage import Reader, Writer
from models import NN1Model, NN2Model
from tasks import TrainingTask, ActivationExtractionTask
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
                NN2Model('CNNH1', user_ids),
                NN1Model('CNNP1', user_ids),
                NN1Model('CNNP2', user_ids),
                NN1Model('CNNP3', user_ids),
                NN1Model('CNNP4', user_ids),
                NN1Model('CNNP5', user_ids),
                NN1Model('CNNP6', user_ids)
                ]

        for model in self.models:
            model.load()

    def compute_score(self, user_id, image):
        """
        Retrieves the similarity score of the given image
        with the given user_id

        Note: Image isn't persisted.
        To persist an image call add_image(user_id, image)

        params:
        user_id: string or int
        image: either a string of the filename or numpy array of an RGB image
        """
        if not self.model:
            raise Exception('Model isn\'t loaded. Call load_model()')

        im = Image(image)
        return self.model.score(user_id, im)

    def get_highest_score_user(self, image):
        """
        Retrieves the user_id with the
        highest similiarity score with the image.

        Note: Image isn't persisted.
        To persist an image call add_image(user_id, image)

        params:
        image: either a string of the filename or numpy array of an RGB image
        """
        if not self.model:
            raise Exception('Model isn\'t loaded. Call load_model()')

        im = Image(image)
        return self.model.get_highest_score_user(im)

    def train(self):
        """
        Trains the model on all images that are currently in storage
        """
        user_ids = self.reader.get_user_ids()
        images = []
        for user_id in user_ids:
            images.append(self.reader.get_images(user_id))
        # Data augmentation
        cloned_images = dp.clone(images, 1)
        reflected_images = ip.apply_reflection(cloned_images)
        images = dp.merge(images, reflected_images)
        images = dp.clone(images, 2)
        images = ip.apply_noise(images)
        data = dp.create_training_data_for_mmdfr(images)
        data_h1, data_p1, data_p2, data_p3, data_p4, data_p5, data_p6, data_y = data

        """
        tasks = [
                TrainingTask(NN2Model, data_h1, data_y, user_ids, 'CNNH1'),
                TrainingTask(NN1Model, data_p1, data_y, user_ids, 'CNNP1'),
                TrainingTask(NN1Model, data_p2, data_y, user_ids, 'CNNP2'),
                TrainingTask(NN1Model, data_p3, data_y, user_ids, 'CNNP3'),
                TrainingTask(NN1Model, data_p4, data_y, user_ids, 'CNNP4'),
                TrainingTask(NN1Model, data_p5, data_y, user_ids, 'CNNP5'),
                TrainingTask(NN1Model, data_p6, data_y, user_ids, 'CNNP6')
                ]

        task_manager = TaskManager(tasks)
        task_manager.run_tasks()

        """
        tasks = [
                ActivationExtractionTask(NN2Model, 'CNNH1', data_h1, user_ids),
                ActivationExtractionTask(NN1Model, 'CNNP1', data_p1, user_ids),
                ActivationExtractionTask(NN1Model, 'CNNP2', data_p2, user_ids),
                ActivationExtractionTask(NN1Model, 'CNNP3', data_p3, user_ids),
                ActivationExtractionTask(NN1Model, 'CNNP4', data_p4, user_ids),
                ActivationExtractionTask(NN1Model, 'CNNP5', data_p5, user_ids),
                ActivationExtractionTask(NN1Model, 'CNNP6', data_p6, user_ids),
                ]

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

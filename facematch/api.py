import consts
import image_processing as ip

from image import Image
from storage import Reader, Writer
from models import NN2Model
from tasks import TrainingTask
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
        self.model_name = consts.NN2
        self.model = NN2Model(None, user_ids, self.model_name)
        self.model.model = self.reader.get_model(self.model_name)

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
        cloned_images = ip.clone_images(images, 1)
        reflected_images = ip.apply_reflection(cloned_images)
        images = ip.merge(images, reflected_images)
        images = ip.clone_images(images, 2)
        images = ip.apply_noise(images)

        tasks = [TrainingTask(NN2Model, images, user_ids, consts.NN2)]
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

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
        self.model = self.reader.get_model(consts.nn2)

    def compute_score(self, user_id, image):
        """
        Note: Image isn't persisted.
        To persist an image call add_image(user_id, image)

        params:
        user_id: string or int
        image: either a string of the filename or numpy array of an RGB image
        """
        im = Image(image)
        return self.model.score(user_id, im)

    def train(self):
        """
        Trains the model on all images that are currently in storage
        """
        ids = self.reader.get_user_ids()

        images = []
        for user_id in ids:
            images.append(self.reader.get_images(user_id))

        # Data augmentation
        reflected_images = ip.clone_images(images)
        ip.apply_reflection(reflected_images)
        images = ip.merge(images, reflected_images)
        ip.apply_cloning(images, 2)
        ip.apply_noise(images)

        tasks = [TrainingTask(NN2Model, images, user_ids)]
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

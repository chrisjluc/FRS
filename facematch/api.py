import consts
import image_processing as ip

from image import Image
from storage import Reader, Writer
from models import NN2Model

class API(object):

    def __init__(self):
        self.reader = Reader()
        self.writer = Writer()

        self.model = self.reader.get_model(consts.nn2)

    def compute_score(id, image):
        im = Image(image)
        score = self.model.predict(im)
        # TODO: process score
        return score

    def train():
        ids = self.reader.get_ids()

        images = []
        for id in ids:
            images.append(self.reader.get_images(id))

        reflected_images = ip.clone_images(images)
        ip.apply_reflection(reflected_images)
        images = ip.merge(images, reflected_images)
        ip.apply_noise(images)

        self.model = NN2Model(images, ids)
        self.model.train()
        self.writer.save_model(self.model)

    def add_image(id, image):
        im = Image(image)
        self.writer.save_image(id, im)

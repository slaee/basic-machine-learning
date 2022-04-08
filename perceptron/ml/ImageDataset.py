
import os
from PIL import Image
import numpy as np

class ImageDataset:
    def __init__(self, path="ml/img-datasets", transform=None):
        self.path = path
        self.transform = transform
        # set data to empty numpy array
        self.data = []
        self.labels = []

    def to2Dflatten(self):
        for filename in os.listdir(self.path):
            img = Image.open(os.path.join(self.path, filename))
            img = img.convert('L')
            data_img = np.array(img)
            data_img = np.where(data_img > 0, 0, 1)
            data_img = data_img.flatten()
            self.data.append(data_img)
            if filename[0] == '1':
                self.labels.append(1)
            else:
                self.labels.append(0)

    def dataX(self):
        return self.data

    def labelY(self):
        return self.labels
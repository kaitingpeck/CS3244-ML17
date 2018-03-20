from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2


list_of_imgs = []
def load_images():
    img_dir = "../data/dataset-resized/"
    for img in os.listdir("."):
        img = os.path.join(img_dir, img)
        if not img.endswith(".jpg"):
            continue
        a = cv2.imread(img)
        if a is None:
            print("Unable to read image", img)
            continue
        list_of_imgs.append(a.flatten())
    train_data = np.array(list_of_imgs)
    print(train_data)
    return train_data

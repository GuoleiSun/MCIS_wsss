from PIL import Image
import cv2
import os
import numpy as np
from os.path import exists

data_path = 'pseudo_label_folder'
save_path=data_path
lists = os.listdir(data_path)
if not exists(save_path):
    os.mkdir(save_path)
for index in lists:
    I = Image.open(data_path + index)
    I = np.array(I, dtype=np.uint8)
    cv2.imwrite(save_path + index, I)

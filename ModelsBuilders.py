import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detecto.utils import read_image 
from detecto.core import Dataset
from detecto.visualize import show_labeled_image 
from detecto.core import DataLoader, Model




#
# dataset_for_model1_path = ""
# dataset1 = Dataset(dataset_for_model1_path)
# model1 = Model(['guitar_bar'])
# model1.fit(dataset1, epochs=3)
#
#
# dataset_for_model2_path = ""
# dataset2 = Dataset(dataset_for_model2_path)
# model2 = Model(['guitar_neck_cropped'])
# model2.fit(dataset2, epochs=3)

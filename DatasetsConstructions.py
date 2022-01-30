import os
import numpy as np
import matplotlib.pyplot as plt
from detecto.utils import read_image
import cv2
from detecto.core import Dataset
from detecto.visualize import show_labeled_image
from detecto.core import DataLoader, Model
import imageio
from UtilsPreprocess import get_window, set_image_to_uint8, align_image, load_detecto_model1, filter_only_images
from ImageManipulations import add_left_hand_skelaton

#################### GUITAR BAR DETECTION, STEP 1 ####################
def build_folder_of_windows_step1(model, src_location, dst_location):
    """
     For detection the chords played on the guitar we want the input
     for the model to be a picture of only the guitar bar.
     That is done in 2 steps, the first step is this one, which we take
     a trained model to detect the general area around a guitar bar, we
     then pass through all the images in src_location and detect the
     area around the guitar bar, crop it, we then rotate that area so
     that the guitar bar will lay flat with the neck horizontal to the
     x-axis and eventually save it in src_destination

    :param model: a detecto model for detecting a guitar_neck from a
     picture of a person playing the guitar to pass the images through
    :param src_location: The location of an images folder (only images
     from which we'll take the images
    :param dst_location: the destination of the model's outputs,
     including the folder name. dst_location="folder_location/folder_name"
    :return:
    """
    files = os.listdir(src_location)
    files = filter_only_images(files)
    os.mkdir(dst_location)

    for i, filename in enumerate(files):
        filename_path = os.path.join(src_location, filename)
        print('filename_path: ', filename_path)
        img = cv2.imread(filename_path)
        window = get_window(img, model)
        # Let's straighten to window

        if window is None:
            continue
        cv2.imwrite(os.path.join(dst_location, "window"+str(i)+".png"), window)


#################### GUITAR BAR DETECTION, STEP 2 ####################
def build_folder_of_windows_step2(model,src_location, dst_location, output_dims=None):
    """
     For detection the chords played on the guitar we want the input
     for the model to be a picture of only the guitar bar.
     That is done in 2 steps, this is the second step, which we take
     a trained model to detect the guitar bar from a picture of a small
     region around the guitar bar. Eventually we'll save all the images
     in src_location in src_destination
    :param model: a 'detecto' model to detect the guitar bar in a
     precise matter
    :param src_location: path to a folder with only images of a cropped
     window around a guitar neck, where the guitar neck is laying flat
     on the x-axis
    :param src_destination: the destination of the model's outputs,
     including the folder name. src_destination="folder_location/folder_name"
    :param output_dims: the dimension of the output images in (w,h)
    :return:
    """
    files = os.listdir(src_location)
    files = filter_only_images(files)
    os.mkdir(dst_location)
    for i, filename in enumerate(files):
        filename_path = os.path.join(src_location, filename)
        img = cv2.imread(filename_path)
        img, line_image = align_image(img)

        window = get_window(img, model)
        if not output_dims is None:
            window = cv2.resize(window, output_dims)

        if window is None:
            continue
        cv2.imwrite(os.path.join(dst_location, "guitar_bar"+str(i)+".png"), window)


##################### GUITAR BAR DETECTION, STEP 1+2 ###################
def build_folder_of_windows(model1, model2, src_location, dst_location, output_dims=None, hand_skelaton_func=None):
    """
    Does build_folder_of_windows_step1 + build_folder_of_windows_step2 together
    :param model1:
    :param model2:
    :param src_location:
    :param dst_location:
    :param output_dims: the dimension of the output images in (w,h)
    :return:
    """
    files = os.listdir(src_location)
    files = filter_only_images(files)
    os.mkdir(dst_location)

    for i, filename in enumerate(files):
        filename_path = os.path.join(src_location, filename)
        img = cv2.imread(filename_path)
        window = get_window(img, model1)

        # Let's straighten to window
        window, line_image = align_image(window)

        if hand_skelaton_func is not None:
            # cv2.imshow("before skelaton", window)
            window = hand_skelaton_func(window)
            cv2.imwrite(os.path.join(dst_location, "window" + str(i) + "_before_crop.png"),window)
            #cv2.imshow("after skelaton", window)
            #cv2.waitKey(0)

        window = get_window(window, model2)
        if window is None:
            continue
        if not output_dims is None:
            window = cv2.resize(window, output_dims)
        cv2.imwrite(os.path.join(dst_location, "window"+str(i)+".png"), window)



### testing:::
#print(os.listdir())
model1 = load_detecto_model1(os.path.join("models", "model1.pth"))
model2 = load_detecto_model1(os.path.join("models", "model2.pth"))
# build_folder_of_windows_step1(model1, os.path.join("datasets","dataset_for_guitar_bar_detection"),
#                               os.path.join("datasets", "temp_folder"))
# build_folder_of_windows_step2(model2, os.path.join("datasets", "dataset_for_guitar_bar_cropping"),
#                               os.path.join("datasets", "output_with_skelaton1"), hand_skelaton_func=add_left_hand_skelaton)


# build_folder_of_windows(model1, model2, os.path.join("datasets","dataset_for_guitar_bar_detection"),
#                         os.path.join("datasets", "ready_images_with_skelaton"), output_dims=(512, 128), hand_skelaton_func=add_left_hand_skelaton)

TEMP_FOLDER_PATH = "datasets/test_folder_for_drive_temp"

print('os: ', os.listdir(TEMP_FOLDER_PATH))
for folder in os.listdir(TEMP_FOLDER_PATH):
    folder_path = os.path.join(TEMP_FOLDER_PATH, folder)
    new_folder_path = os.path.join("datasets","test_folder_for_drive_temp_post_step1", folder)
    print('folder_path: ', folder_path)
    print('new_folder_path: ', new_folder_path)
    #os.mkdir(new_folder_path)
    build_folder_of_windows_step1(model1, folder_path, new_folder_path)

#
#
# im = cv2.imread("datasets/Roskin_dataset_by_chord/A/A_chord_video_1_116.jpg")
# cv2.imshow('image', im)
# cv2.waitKey(0)
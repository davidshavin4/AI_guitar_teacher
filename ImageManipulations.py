# This file will contain different functions to manipulate the images
# so that we'll try different ways of inserting the data into our
# model.

import cv2
import mediapipe as mp
import time
import os




#cap = cv2.VideoCapture(0)


def add_left_hand_skelaton(image):
    """

    :param image: image in BGR format
    :return:
    """
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLms,
                                  mpHands.HAND_CONNECTIONS)
    return image
#
#
# IMAGES_PATH = "datasets/dataset_for_guitar_bar_cropping"
#
# for filename in os.listdir(IMAGES_PATH):
#     if filename.split('.')[1] != 'png':
#         continue
#     print('filename: ', filename)
#     file_path = os.path.join(IMAGES_PATH, filename)
#     im = cv2.imread(file_path)
#     print('im shape: ', im.shape)
#     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#     imgRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             mpDraw.draw_landmarks(im, handLms, mpHands.HAND_CONNECTIONS)
#
#     cv2.imshow("guitar bar", im)
#     cv2.waitKey(0)
#
#
# # while True:
# #     success, img = cap.read()
# #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     results = hands.process(imgRGB)
# #
# #     if results.multi_hand_landmarks:
# #         for handLms in results.multi_hand_landmarks:
# #             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
# #
# #     cv2.imshow("Image", img)
# #     cv2.waitKey(1)

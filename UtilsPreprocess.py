import cv2
import numpy as np
from detecto.core import Model

def load_detecto_model1(model_path):
    """
     This function loads model1, which is the model that detects
     the guitar bar from a general picture of a person playing
     the guitar
    :param model_path: in format: "model_location/model_name"
    :return: the model
    """
    model = Model.load(model_path, ['guitar_bar'])
    return model

def load_detecto_model2(model_path):
    """
     This function loads model1, which is the model that detects
     the guitar bar exactly from a window of the guiar bar
    :param model_path: in format: "model_location/model_name"
    :return: the model
    """
    model = Model.load(model_path, ['guitar_neck_cropped'])
    return model

def set_image_to_uint8(image):
    """
     takes an image with number in what ever range and fits them to
     the range 0-255 with dtype uint-8
    :param image: Image to be fitted
    :return: fitted image
    """
    fitted_image =(((image-np.min(image, axis=(0,1)))) /
            (np.max(image, axis=(0,1)) - np.min(image, axis=(0,1))) *
            255).astype(np.uint8)
    return fitted_image

def normalize(image):
    """
    image in numpy format: shape=(h,w,3)
    :param image:
    :return:
    """
    image = image.astype(np.float32)
    mean = np.mean(image, axis=(0,1))
    var = np.mean(image**2, axis=(0,1)) - mean**2
    return (image-mean) / var**0.5


def torch_to_numpy(image):
    """
     This function will transform an image in the format of a tensor,
     meaning: shape=() to the format of numpy: shape=()
    :param im: image in torch format
    :return: image in numpy format
    """

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def align_image(image):
    """
     This function calculates the lines contained in the image using
     the Hough transform and then aligns the image accordingly
    :param im: a colored image in numpy format. type=u-int8
    :return: list: [an aligned image, the image will all the detected
     lines for the user to get a better understanding why the picture
     was rotated the way it was]
    """
    rho, threshold, min_line_length, max_line_gap = 1, 15, 50, 20
    line_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_8U, 0, 1, 3)
    idx_white, idx_black = edges>127, edges<=127
    edges[idx_white] = 255
    edges[idx_black] = 0

    # Run Hough transform on edges to detect lines:
    lines = cv2.HoughLinesP(edges, rho, np.pi / 180, threshold,
                            np.array([]), min_line_length,
                            max_line_gap)
    slopes = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = abs((y2 - y1) / (x2 - x1))
            slopes.append(slope)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    median_slope = np.median(slopes)
    angle = np.arctan(median_slope) * (180 / np.pi)
    rotated_image = rotate(image, -angle)
    return rotated_image, line_image



def get_window(image, model):
    """


     NOTE: if image is not tensor, then it should be loaded with cv2 and
     not with imageio
    :param image:
    :param model:
    :return:
    """
    targets = model.predict(image)
    if len(targets[0]) == 0:
        return None
    xmin, ymin, xmax, ymax = targets[1][0]
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    # print("xmin, ymin, xmax, ymax: ", xmin, ymin, xmax, ymax)
    if isinstance(image, np.ndarray):
        return image[ymin:ymax, xmin:xmax,:]
    return image[:, ymin:ymax, xmin:xmax]

def filter_only_images(lst):
    """

    :param lst: a list of filenames
    :return: a list of filenames with only a jpg, jpeg, JPG, png, PNG
    """
    images_format = {'jpg', 'jpeg', 'JPG', 'png', 'PNG'}
    output_list = []
    for filename in lst:
        _, format = filename.split('.')
        if format in images_format:
            output_list.append(filename)
    return output_list
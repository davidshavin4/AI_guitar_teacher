import cv2
import mediapipe as mp
import numpy as np
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,
                 trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon,
                                        self.trackCon)
        # self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw_on_image=True):
        """

        :param img: image in BGR format (read by cv2)
        :return:
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if draw_on_image:
            h,w,c = img.shape
            #print('here1')
            #print('results: ', self.results.multi_hand_landmarks)
            if self.results.multi_hand_landmarks:
                #print('here2')
                for handLms in self.results.multi_hand_landmarks:
                    # for i, lm in enumerate(handLms.landmark):
                    #     cx, cy = int(lm.x * w), int(lm.y * h)
                    #     if i==8:
                    #         cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    #print('here')
        return img

    def findPositions(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for i, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([i, cx, cy])
                if i == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                if i == 12:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                               cv2.FILLED)
                if i == 16:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255),
                               cv2.FILLED)
        return lmList


def detect_chord(lms):
   # print("lms: ", lms)

    if len(lms)==21:
        print("finger y position 2,3,4: ", lms[8][2], lms[12][2], lms[16][2])
        if lms[16][2] < lms[12][2] < lms[8][2]:
            print("C")
        elif lms[8][2] < lms[12][2] < lms[16][2]:
            print("A")
        elif lms[12][2] > lms[8][2] > lms[16][2]:
            print("Dm")
        else:
            print("try different chord..")


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




def video_capture(handDetector):

    #cap = cv2.VideoCapture(0) # if you want to use webcam
    cap = cv2.VideoCapture('let_it_be_long.mp4') # if you want to use video in folder
    while True:
        success, img = cap.read()

        imgLms = handDetector.findHands(img, True)
        lmsList = handDetector.findPositions(img, 0)


        ######### align image ##########

        output = align_image(imgLms)
        if output:
            rotated_image, line_image = output
            #img = line_image
            img = rotated_image
        #print('aligned_image shape: ', aligned_image)

        ################################

        # print('lmList: ', lmsList)
        detect_chord(lmsList)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    handDetector = handDetector()
    video_capture(handDetector)
    # im = cv2.imread("C_003.jpg")
    # im = cv2.resize(im, (1024, 512))
    #
    # imLms = handDetector.findHands(im, True)
    # cv2.imshow("C chord", imLms)
    #
    # if cv2.waitKey(0):
    #     cv2.destroyAllWindows()

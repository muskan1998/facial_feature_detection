#python mouth.py --shape-predictor shape_predictor_68_face_landmarks.dat --image 1.jpg


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect faces in the grayscale image
rects = detector(gray, 1)

for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
	shape = predictor(gray, rects[0])
	shape = face_utils.shape_to_np(shape)
	(j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
	pts = shape[j:k]
	if name == "mouth":
		hull = cv2.convexHull(pts)
		cv2.drawContours(image, [hull], -1, (163, 38, 32), -1)

cv2.imshow("Result", image)
cv2.waitKey(0)

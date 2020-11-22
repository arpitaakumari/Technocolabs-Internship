# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:05:20 2020

@author: Arpita Kumari

FACE MASK DETECTOR PROJECT @ Technocolabs
"""



#importing the libraries
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

#face model
prototxt = "F:/projects/Technocolabs/deploy.prototxt"
weights = "F:/projects/Technocolabs/res10_300x300_ssd_iter_140000.caffemodel"
facemodel = cv2.dnn.readNet(prototxt, weights)

#specifying the font
font = cv2.FONT_HERSHEY_PLAIN

# load the face mask detector model
maskmodel = tf.keras.models.load_model("F:/projects/Technocolabs/model.h5")


#function to both face and mask detection
def prediction(frame, facemodel, maskmodel):
    
	# grab the dimensions of the frame and then construct a blob of it
	(h, w) = frame.shape[:2]
    
    #blob = binary large object
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network of face detection
	facemodel.setInput(blob)
	detections = facemodel.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskmodel.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


print(" Opening the video file ")
#cap = cv2.VideoCapture("F:/projects/Facemask Detector/video5.mp4")


def camera_stream(frame):
   # cap = cv2.VideoCapture(0)
   frame = cv2.flip(frame, 1, 1)
   (locs, preds) = prediction(frame, facemodel, maskmodel)

   for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        #take screenshot of the people with no mask
        if label == "No Mask":
            cv2.imwrite("F:/projects/Technocolabs/nomaskss/ss.png", frame[startY:endY, startX:endX])

		# display the label and bounding box rectangle on the output
		# frame
        cv2.rectangle(frame,(startX, startY),(endX,endY),color,2)
        cv2.putText(frame, label, (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
 
   #return cv2.imencode('.jpg', frame)[1].tobytes()
   return frame
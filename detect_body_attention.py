# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import glob
from tqdm import tqdm

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input_folder", required=False,
		default="./output-final/image_body",
		help="path to input folder image")
	ap.add_argument("-o", "--output_folder", required=False,
		default="./output_cnn_",
		help="path to input folder image")
	ap.add_argument("-f", "--face", type=str,
		default="./models/face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="./models/mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.55,
		help="minimum probability to filter weak detections")
	ap.add_argument("-a", "--ananymous", type=bool, default=False,
		help="make all output pictures anonymous")
	args = vars(ap.parse_args())

	dirs = ['image_body_attention','image_body_attention_with_box']
    
    # check output_folder exists
	if not os.path.exists(args['output_folder']):
		os.makedirs(args['output_folder'])

	for folder in dirs:
		if not os.path.exists(os.path.sep.join([args["output_folder"], folder])):
			os.makedirs(os.path.sep.join([args["output_folder"], folder]))

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# we fetch images path
	images_path = glob.glob(os.path.sep.join([args["input_folder"], "*"]))
	if len(images_path)==0:
		print("No pics in the forlder")

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	for image_path in tqdm(images_path):
		image_name = image_path.split(os.path.sep)[-1]
		image = cv2.imread(image_path)
		orig = image.copy()
		(h, w) = image.shape[:2]

		# construct a blob from the image
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detection
		net.setInput(blob)
		detections = net.forward()
		
		# we count face detected in one pic
		count = 0

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				count = count+1
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
				face = image[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))

				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis=0)

				# pass the face through the model to determine if the face
				# has a mask or not
				(mask, withoutMask) = model.predict(face)[0]

				# determine the class label and color we'll use to draw
				# the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(image, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

		# we keep picture if one face only detected
		if count==1:
			# we write the original image inside the output_folder
			cv2.imwrite(f"{args['output_folder']}/image_body_attention/{image_name}", image)
		if count>1:
			cv2.imwrite(f"{args['output_folder']}/image_body_attention_with_box/{image_name}", image)

		# show the output image
		# cv2.imwrite(f"{args['output_folder']}/image_body_attention_with_box/{image_name}", image)
	
if __name__ == "__main__":
	mask_image()
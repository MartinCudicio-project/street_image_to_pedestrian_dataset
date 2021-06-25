# import the necessary packages
import cv2
import os
import glob
from tqdm import tqdm
import imutils

def face_detection(input_folder,output_folder,anonymous,logs):
		
	# we create folders if not exist
	# image_body_facebox for face detection analysis
	# image_body_final for final images
	if logs==True:
	# subfolders needed
		dirs = ['image_body_final','image_body_facebox']
	else:
		dirs = ['image_body_final']

	# check output_folder exists
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	for folder in dirs:
		if not os.path.exists(os.path.sep.join([output_folder, folder])):
			os.makedirs(os.path.sep.join([output_folder, folder]))


	# load the face mask detector model from disk
	print("[INFO] loading face detector model...")
	face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalcatface_extended.xml')
	# param model
	width_wanted=400
	scale=1.1

	# we fetch images path
	images_path = glob.glob(os.path.sep.join([input_folder, "*"]))
	if len(images_path)==0:
		print("No pics in the forlder")

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	for image_path in tqdm(images_path):
		image_name = os.path.splitext(os.path.basename(image_path))[0]
		image_extension = os.path.splitext(os.path.basename(image_path))[1]
		image = cv2.imread(image_path)
		orig = image.copy()

		# resize with width_wanted for analysis
		image_resized = imutils.resize(image, width=width_wanted)
		ratio = image.shape[0]/image_resized.shape[0]
		img_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

		# detect faces
		faces = face_cascade.detectMultiScale(img_gray, scale, 3)
		if len(faces) == 1 :
			faces = (faces*[[ratio]]).astype(int)
			# Draw rectangle around the faces
			for (x, y, w, h) in faces:
				cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)
				
				face = orig[y:y+h, x:x+w]
				if anonymous==True:				
					# apply a gaussian blur on this new recangle image
					face = cv2.GaussianBlur(face,(23, 23), 30)
					# overlay this blurry rectangle to our final image
					orig[y:y+h, x:x+w] = face
				cv2.imwrite(f"{output_folder}/image_body_final/{image_name}{image_extension}", orig)
		
		if logs==True:
			cv2.imwrite(f"{output_folder}/image_body_facebox/{image_name}{image_extension}", image)
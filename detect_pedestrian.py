import numpy as np
import cv2
import os 
import glob as glob
from tqdm import tqdm
import argparse
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def photo_body(input_folder,output_folder,logs):
    
    # we create folder if not exist
    # image_full_with_box for raw_image with box detection
    # image_body for roi pedestrian
    if logs==True:
        dirs = ['image_body','image_body_box']
    else:
        dirs = ['image_body']
    
    # check output_folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in dirs:
        if not os.path.exists(os.path.sep.join([output_folder, folder])):
            os.makedirs(os.path.sep.join([output_folder, folder]))

    # param model
    width_wanted=400
    scale=1.13
    winStride =(4, 4)
    padding=(8, 8)
    
    # we fetch images path
    images_path = glob.glob(os.path.sep.join([input_folder, "*"]))
    if len(images_path)==0:
        print("No pics in the forlder")

    for image_path in tqdm(images_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_extension = os.path.splitext(os.path.basename(image_path))[1]
        # count number poeple inside 1 image 
        count=0
        
        # read image
        image = cv2.imread(image_path)
        orig = image.copy()
        
        # resize with width_wanted for analysis
        image_resized = imutils.resize(image, width=width_wanted)
        ratio = image.shape[0]/image_resized.shape[0]
        img_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        # detect pedestrians
        rects, weights = hog.detectMultiScale(img_gray, winStride=winStride, padding=padding, scale=scale)
        if len(rects)>0:
            rects = (rects*[[ratio]]).astype(int)
            for i, (x, y, w, h) in enumerate(rects):
                if weights[i] < 0.13:
                    continue
                elif weights[i] < 0.3 and weights[i] > 0.13:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), int(ratio))
                if weights[i] < 0.9 and weights[i] > 0.3:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (50, 122, 255), int(ratio))
                if weights[i] > 0.9:
                    count=count+1
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), int(ratio))
                    cv2.imwrite(f"{output_folder}/image_body/{image_name}_{count}{image_extension}", orig[y:y+h,x:x+w])

        # write images if logs option activated
        if logs==True:
            cv2.imwrite(f"{output_folder}/image_body_box/{image_name}{image_extension}", image)
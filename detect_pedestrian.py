import numpy as np
import cv2
import os 
import glob as glob
from tqdm import tqdm
import argparse
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def photo_body():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_folder", required=False,
        default="./raw_images",
        help="path to input folder image")
    ap.add_argument("-o", "--output_folder", required=False,
        default="./output",
        help="path to input folder image")
    args = vars(ap.parse_args())
    
    # we create folder if not exist
    # image_full for raw_image with box detection
    # image_body for roi pedestrian
    dirs = ['image_body','image_full_with_box']
    
    # check output_folder exists
    if not os.path.exists(args['output_folder']):
        os.makedirs(args['output_folder'])

    for folder in dirs:
        if not os.path.exists(os.path.sep.join([args["output_folder"], folder])):
            os.makedirs(os.path.sep.join([args["output_folder"], folder]))

    # param model
    width_wanted=400
    scale=1.13
    winStride =(4, 4)
    padding=(8, 8)
    
    # we fetch images path
    images_path = glob.glob(os.path.sep.join([args["input_folder"], "*"]))
    if len(images_path)==0:
        print("No pics in the forlder")

    # images_done = [w.replace('./outputs/image_full',input_path) for w in glob.glob('./outputs/image_full/*')]
    # images_final = list(set(images_path)-set(images_done))
    for image_path in tqdm(images_path):
        image_name = image_path.split(os.path.sep)[-1]
       
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
                    cv2.imwrite(f"{args['output_folder']}/image_body/{count}_{image_name}", orig[y:y+h,x:x+w])

             
        cv2.imwrite(f"{args['output_folder']}/image_full_with_box/{image_name}", image)

if __name__ == "__main__":
	photo_body()

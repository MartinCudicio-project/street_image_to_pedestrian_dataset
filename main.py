import argparse
import os

# from detect_body_attention import mask_image
from detect_body_attention_haar import face_detection
from detect_pedestrian import photo_body

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_folder", required=False,
        default="./raw_images",
        help="path to input folder image")
    ap.add_argument("-o", "--output_folder", required=False,
        default="./output",
        help="path to input folder image")
    ap.add_argument("-l", "--logs", type=bool, default=False,
        help="create dir with transformation images for each step")
    ap.add_argument("-a", "--anonymous", type=bool, default=False,
        help="make all output pictures anonymous")
    args = vars(ap.parse_args())
    
    print(args)
    print("-------------------------------------------------------")
    print("STEP 1 -- Pedestrian Detection")
    photo_body(args["input_folder"],args['output_folder'],args['logs'])

    print()
    print("-------------------------------------------------------")
    print("STEP 2 -- Pedestrian Attention")
    # we set input_folder to output_folder/image_body
    args['input_folder'] = os.path.sep.join([args["output_folder"],'image_body'])
    face_detection(args["input_folder"],args["output_folder"],args["anonymous"],args['logs'])

    
	
if __name__ == "__main__":
	main()
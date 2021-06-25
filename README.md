# Street image to cleaned pedestrian dataset

this project is an image transformation pipeline. The goal is to go from raw street images to cleaned pedestrian dataset images.

there are 2 steps of transformation: 
 1. pedestrian detection
 2. detection of pedestrian attention with face


### prerequisite

- Anaconda

## Get started

Create new anaconda env with the provided environment file
`conda env create -f environment.yml` 

## Made with

- openCV
- TensorFlow
- haarcascade-frontalface

## Usage

Don't forget to activate anaconda environment

`conda activate detection_env`

`python main.py`

agrs : 

- `--input_folder` : path to the input folder (default: raw_images/)
- `--output_folder`: path to the output folder (default: output/)
- `--anonymous` : make output images anonymous (default: False)
- `logs`: create subfolder at each step to display images transformation (default: False)

## Auteurs

* **Martin Cudicio** _alias_ [@MartinCudicio-project](https://github.com/outout14)

Read the list of [contributors](https://github.com/MartinCudicio-project/street_image_to_pedestrian_dataset/contributors) to see who helped the project!

## License

This project is licensed under the `MIT` license

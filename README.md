# Duplicate Image Deletion

This script is used to compare a series of images in a given folder and delete duplicate images based on contour analysis and a threshold value. The script performs the following steps:

1. Reads a folder path and parameters from the command line.
2. Preprocesses the images by applying a color mask and Gaussian blur.
3. Compares each image with the previous one using contour analysis.
4. Calculates a score based on contour areas and compares it to a threshold value.
5. Deletes the base image if the score is below the threshold.
6. Moves to the next image and repeats the process.

## Usage

python solution.py input_folder_path min_contour_area threshold

Run the script with the desired parameters as shown above

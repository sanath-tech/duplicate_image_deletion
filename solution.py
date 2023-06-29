import os
import argparse
import cv2
import imutils
import shutil
from datetime import datetime

# Function to draw a color mask on an image based on given borders and color
def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


# Function to preprocess an image for change detection
def preprocess_image_change_detection(
    img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)
):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


# Function to compare two frames for change detection
def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


# Custom sorting key function to sort image names based on timestamp
def get_timestamp(filename):
    if "-" in filename:
        return int(filename.split("-")[1].split(".")[0])  # Unix timestamp
    else:
        timestamp = filename.split("_")[1].replace(
            "__", "_"
        )  # c%id%_yyyy_mm_dd__hh__mm__ss format
        return int(timestamp.replace("_", ""))


if __name__ == "__main__":

    # Parse command line argument
    parser = argparse.ArgumentParser(
        description="Read folder path, Gaussian blur filter radius, min contour area, and threshold from command line."
    )
    parser.add_argument("input_folder_path", type=str, help="Path to the input folder")
    parser.add_argument(
        "min_contour_area",
        type=int,
        help="Minimum contour area",
        nargs="?",
        default=3000,
    )
    parser.add_argument(
        "threshold", type=int, help="Threshold", nargs="?", default=4025
    )
    args = parser.parse_args()

    folder_path = args.input_folder_path
    min_contour_area = args.min_contour_area
    threshold = args.threshold

    # Initialize an empty list to store the image names
    image_names = []

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # Check if the file is a PNG image
            image_names.append(filename)

    # Sort the image_names list based on the timestamp
    image_names = sorted(image_names, key=get_timestamp)

    base_image_address = os.path.join(folder_path, image_names[0])
    base_image = cv2.imread(base_image_address)

    # Process each image starting from the second image
    for i in range(1, len(image_names)):
        current_image = cv2.imread(os.path.join(folder_path, image_names[i]))

        # Preprocess the base image
        if base_image is not None:
            base_image = cv2.resize(base_image, (640, 480))
            base_image_processed = preprocess_image_change_detection(
                base_image, gaussian_blur_radius_list=[3]
            )

        # Preprocess the current image
        if current_image is not None:
            current_image = cv2.resize(current_image, (640, 480))
            current_image_processed = preprocess_image_change_detection(
                current_image, gaussian_blur_radius_list=[3]
            )

        score = compare_frames_change_detection(
            base_image_processed,
            current_image_processed,
            min_contour_area=min_contour_area,
        )[0]

        if score < threshold:
            # Delete the base duplicate image
            os.remove(base_image_address)

        # Update the base image and its address
        base_image = current_image
        base_image_address = os.path.join(folder_path, image_names[i])

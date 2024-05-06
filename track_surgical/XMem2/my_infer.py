import os
import random
from inference.run_on_video import run_on_video
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def calculate_iou(ground_truth_folder, prediction_folder):
    filenames = os.listdir(ground_truth_folder)
    iou_values = []

    for filename in filenames:
        # Read the ground truth mask (grayscale image)
        ground_truth_mask = cv2.imread(os.path.join(ground_truth_folder, filename), cv2.IMREAD_GRAYSCALE)
        # print("ground_truth_mask.shape",ground_truth_mask.shape)
        
        # Read the corresponding prediction mask (grayscale image)
        prediction_mask = cv2.imread(os.path.join(prediction_folder, filename), cv2.IMREAD_GRAYSCALE)
        # print("prediction_mask.shape", prediction_mask.shape)
        # Check if the masks have the same shape
        if ground_truth_mask.shape != prediction_mask.shape:

            # Reshape image1 to match the shape of image2
            prediction_mask = cv2.resize(prediction_mask, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]))

        # Threshold the masks to obtain binary masks
        _, binary_ground_truth = cv2.threshold(ground_truth_mask, 0, 1, cv2.THRESH_BINARY)
        _, binary_prediction = cv2.threshold(prediction_mask, 0, 1, cv2.THRESH_BINARY)

        # Compute the intersection and union
        intersection = np.logical_and(binary_ground_truth, binary_prediction)
        union = np.logical_or(binary_ground_truth, binary_prediction)

        # Calculate the IoU
        iou = np.sum(intersection) / np.sum(union)
        iou_values.append(iou)

    return iou_values


def calculate_frame_order(images_folder, num_parts):
    # Get the list of image files in the folder
    image_files = [file for file in os.listdir(images_folder) if file.endswith('.jpg') or file.endswith('.png')]
    num_images = len(image_files)
    
    # Calculate the number of images in each part
    images_per_part = num_images // num_parts
    
    # Calculate the frame number list of the first frame of each part
    frame_numbers = []
    for i in range(num_parts):
        # Calculate the frame number of the first image in the current part
        frame_number = i * images_per_part
        
        # Add the frame number to the list
        frame_numbers.append(frame_number)
    
    return frame_numbers



if __name__ == '__main__':
    imgs_path = f"/mnt/data-hdd/jieming/esd_dataset/{dataset}/resized_image"
    masks_path =  imgs_path.replace("resized_image", "sam_pred")

    print(imgs_path, masks_path)
    
    frames_with_masks = [0,]
    output_path = f'/mnt/data-hdd/jieming/XMem2/{dataset}'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    stats = run_on_video(imgs_path, masks_path, output_path, frames_with_masks, compute_iou=True)  #  stats: pandas DataFrame

        
        
    
    
    

import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

def binary_mask(image):
    binary_image = np.where(image != 0, 1, 0)
    return binary_image

def calculate_metrics(output_mask, ground_truth_mask):
    output_mask_binary = binary_mask(output_mask)
    ground_truth_mask_binary = binary_mask(ground_truth_mask)

    intersection = np.logical_and(output_mask_binary, ground_truth_mask_binary)
    union = np.logical_or(output_mask_binary, ground_truth_mask_binary)

    iou = np.sum(intersection) / np.sum(union)
    accuracy = accuracy_score(ground_truth_mask_binary.flatten(), output_mask_binary.flatten())
    dice = 2 * np.sum(intersection) / (np.sum(output_mask_binary) + np.sum(ground_truth_mask_binary))

    return iou, accuracy, dice

output_folder = "/mnt/data-hdd/jieming/XMem2"
esd_dataset_folder = "/mnt/data-hdd/jieming/esd_dataset"

iou_list = []
accuracy_list = []
dice_list = []

for step in range(2, 4):
    output_step_folder = os.path.join(output_folder, f"step{step}/masks")
    esd_step_folder = os.path.join(esd_dataset_folder, f"step{step}")


    for image_file in os.listdir(output_step_folder):
        output_mask = cv2.imread(os.path.join(output_step_folder, image_file), cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = cv2.imread(os.path.join(esd_step_folder, "final_mask", image_file), cv2.IMREAD_GRAYSCALE)

        iou, accuracy, dice = calculate_metrics(output_mask, ground_truth_mask)

        iou_list.append(iou)
        accuracy_list.append(accuracy)
        accuracy_list.append(dice)


print("Mean Metrics for all images in each step:")
print(f"Mean IoU: {np.mean(iou_list)}")
print(f"Mean Accuracy: {np.mean(accuracy_list)}")
print(f"Mean Dice: {np.mean(accuracy_list)}")
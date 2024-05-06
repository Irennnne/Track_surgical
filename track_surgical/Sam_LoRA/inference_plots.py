import torch
import numpy as np 
from src.segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
from src.processor import Samprocessor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image, ImageDraw
import yaml
import json
from torchvision.transforms import ToTensor
import os
import cv2
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
"""
This file is used to plots the predictions of a model (either baseline or LoRA) on the train or test set. Most of it is hard coded so I would like to explain some parameters to change 
referencing by lines : 
line 22: change the rank of lora; line 98: Do inference on train (inference_train=True) else on test; line 101 and 111 is_baseline arguments in fuction: True to use baseline False to use LoRA model. 
"""
sam_checkpoint = "sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = build_sam_vit_b(checkpoint=sam_checkpoint)
rank = 512
sam_lora = LoRA_sam(sam, rank)
sam_lora.load_lora_parameters("/mnt/data-hdd/jieming/tracking/sam/Sam_LoRA/endo17_10epoch_lora_rank512.safetensors")
model = sam_lora.sam

predicted_masks = []
ground_truth_masks = []


def inference_model(sam_model, image_path, filename, output_path=None,mask_path=None, bbox=None, is_baseline=False, dataset = None):
    if is_baseline == False:
        model = sam_model.sam
        rank = sam_model.rank
    else:
        model = build_sam_vit_b(checkpoint=sam_checkpoint)

    model.eval()
    model.to(device)
    image = Image.open(image_path)
    if mask_path != None:
        mask = Image.open(mask_path)
        # mask = mask.convert('1')
        mask = np.array(mask)
        if mask.size != None:
                # Change all non-zero values to 1
            mask[mask != 0] = 1
        ground_truth_mask =  np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
        
    else:
        box = bbox

    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(box),
        multimask_output=False,
    )
    masks = np.array(masks)
    if masks.size != None:
        # Change all non-zero values to 1
        masks[masks != 0] = 1
    print("pred",np.unique(masks))

    # cv2.imwrite(output_path, masks)
    if mask_path == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline ="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")
        ax2.imshow(masks[0])
        if is_baseline:
            ax2.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(f"./plots/endo_baseline/{filename}_baseline.jpg")
        else:
            ax2.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(f"./plots/endo_rank{rank}/{filename[:-4]}_rank{rank}.jpg")

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline ="red", width = 15)
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")
        print("gt_mask", np.unique(ground_truth_mask))
        ax2.imshow(ground_truth_mask)
        ax2.set_title(f"Ground truth mask: {filename}")
        mask = np.array(masks[0])
        if mask.size != None:
                # Change all non-zero values to 1
            mask[mask != 0] = 1
        ax3.imshow(mask)
        
        plt.imsave(output_path, mask, cmap='gray')
        
        predicted_masks.append(mask)
        ground_truth_masks.append(ground_truth_mask)
                    
        vis_path = f"./{dataset}_plot/"
                    
        if not os.path.exists(vis_path):
                os.makedirs(vis_path)
                
        if is_baseline:
            ax3.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(os.path.join(vis_path, f"{filename[:-4]}_baseline.jpg"))
        else:
            ax3.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(os.path.join(vis_path, f"{filename[:-4]}_rank{rank}.jpg"))
        
        


# Open configuration file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# # Open annotation file
# f = open('annotations.json')
# annotations = json.load(f)


# train_set = annotations["train"]
# test_set = annotations["test"]
dataset = "step1"



if dataset == "train":

    for image_name, dict_annot in train_set.items():
        image_path = f"./dataset/train/images/{image_name}"
        inference_model(sam_lora, image_path, filename=image_name, mask_path=dict_annot["mask_path"], bbox=dict_annot["bbox"], is_baseline=False)


if dataset == '18':
    image_folder_path = "/mnt/data-hdd/jieming/endovis18/ISINet_Train_Val/val/images"
    for image_name in tqdm(os.listdir(image_folder_path)):
        image_path = os.path.join(image_folder_path, image_name)
        mask_path =  image_path.replace("images", "annotations")
        output_path = image_path.replace("images", "sam_pred")
        print(image_path, mask_path)
        # inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=True, dataset= dataset)
        inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=False, dataset= dataset)

elif dataset == '17':
    image_folder_path = "/mnt/data-hdd/jieming/endovis17_split/val/images"
    mask_folder_path = "/mnt/data-hdd/jieming/endovis17_split/val/masks"
    for image_name in tqdm(os.listdir(image_folder_path)):
        image_path = os.path.join(image_folder_path, image_name)
        mask_name = image_name.replace(".jpg", ".png")
        mask_path =  os.path.join(mask_folder_path, mask_name)
        output_path = image_path.replace("images", "sam_pred")
        print(image_path, mask_path)
        # inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=True, dataset= dataset)
        inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=False,dataset = dataset)

for dataset in ["step3"]:
    image_folder_path = f"/mnt/data-hdd/jieming/esd_dataset/{dataset}/resized_image"
    count = 0
    for image_name in tqdm(os.listdir(image_folder_path)):
        image_path = os.path.join(image_folder_path, image_name)
        mask_path =  image_path.replace("resized_image", "final_mask")
        output_path = image_path.replace("resized_image", "test")
        print(image_path, mask_path)
        inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=True, dataset = dataset)
        inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=False, dataset = dataset)
        count += 1
        if count == 10:
            break
# dataset_path = "/mnt/data-hdd/jieming/esd_dataset"

# for video_name in ["step1", "step2", "step3"]:
    
#     image_folder_path = os.path.join(dataset_path, video_name, "resized_image")

#     #create a new folder for output and visualization
#     output_path = image_folder_path.replace("resized_image", "saml_output")
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
        
#     vis_path = image_folder_path.replace("resized_image", "saml_vis")
#     if not os.path.exists(vis_path):
#         os.makedirs(vis_path)
        
    
#     for image_name in sorted(os.listdir(image_folder_path)):
#         image_path = os.path.join(image_folder_path, image_name)
#         mask_path =  image_path.replace("resized_image", "final_mask")
#         output_path = image_path.replace("resized_image", "saml_output")
#         print(image_path, mask_path)
#         inference_model(sam_lora, image_path, filename=image_name, output_path=output_path, mask_path=mask_path, bbox=None, is_baseline=False)
#         break




# Calculate mean IoU
mean_iou = 0.0
for i in range(len(predicted_masks)):
    intersection = np.logical_and(ground_truth_masks[i], predicted_masks[i])
    union = np.logical_or(ground_truth_masks[i], predicted_masks[i])
    iou = np.sum(intersection) / np.sum(union)
    mean_iou += iou
mean_iou /= len(predicted_masks)

# Calculate mean accuracy
mean_accuracy = 0.0
for i in range(len(predicted_masks)):
    accuracy = np.mean(ground_truth_masks[i] == predicted_masks[i])
    mean_accuracy += accuracy
mean_accuracy /= len(predicted_masks)

# Calculate mean dice coefficient
mean_dice = 0.0
for i in range(len(predicted_masks)):
    intersection = np.logical_and(ground_truth_masks[i], predicted_masks[i])
    dice = (2.0 * np.sum(intersection)) / (np.sum(ground_truth_masks[i]) + np.sum(predicted_masks[i]))
    mean_dice += dice
mean_dice /= len(predicted_masks)

print("Mean IoU:", mean_iou)
print("Mean Accuracy:", mean_accuracy)
print("Mean Dice:", mean_dice)


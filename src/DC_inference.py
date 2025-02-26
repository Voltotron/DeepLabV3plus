import torch
import argparse
import cv2
import os
import numpy as np
import shutil
from PIL import Image
from torchvision import transforms
from config import ALL_CLASSES, LABEL_COLORS_LIST
from model import prepare_model
from metrics import iou
from utils import get_label_mask, draw_segmentation_map, image_overlay

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help="Path to the trained model (.pth file)")
parser.add_argument('--input', required=True, help="Input directory with images and masks")
parser.add_argument('--output', default="../outputs/inference_results/", help="Output directory to save results")
parser.add_argument('--size', default=256, type=int, help="Resolution size for inference images")
args = parser.parse_args()

# Paths for test images and masks
test_images_path = os.path.join(args.input, "images")
test_masks_path = os.path.join(args.input, "masks")

if not os.path.exists(test_images_path) or not os.path.exists(test_masks_path):
    raise FileNotFoundError("Missing 'images' or 'masks' directory inside the input folder.")

# Clear and create output directory
if os.path.exists(args.output):
    shutil.rmtree(args.output)
os.makedirs(args.output, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = prepare_model(len(ALL_CLASSES)).to(device)
ckpt = torch.load(args.model, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_files = sorted(os.listdir(test_images_path))
mask_files = sorted(os.listdir(test_masks_path))

iou_scores_concrete = []
iou_scores_formwork = []

total_classes = [1, 2] 

print(f"Processing {len(image_files)} images (resized to {args.size}x{args.size})\n")

for i, img_name in enumerate(image_files):
    img_path = os.path.join(test_images_path, img_name)
    mask_path = os.path.join(test_masks_path, mask_files[i])
    output_path = os.path.join(args.output, img_name)

    # Load and resize image
    image = Image.open(img_path).convert("RGB")
    image_resized = image.resize((args.size, args.size))
    image_tensor = transform(image_resized).unsqueeze(0).to(device)

    # Load and resize mask
    mask = Image.open(mask_path).convert("RGB")
    mask_resized = mask.resize((args.size, args.size), resample=Image.NEAREST)
    mask_label = get_label_mask(np.array(mask_resized), list(range(len(ALL_CLASSES))), LABEL_COLORS_LIST)
    mask_tensor = torch.tensor(mask_label, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dim

    # Inference
    with torch.no_grad():
        output = model(image_tensor)['out']

    # Calculate IoU for Concrete and Formwork
    iou_concrete = iou(output, mask_tensor, class_index=1)
    iou_formwork = iou(output, mask_tensor, class_index=2)
    
    iou_scores_concrete.append(iou_concrete)
    iou_scores_formwork.append(iou_formwork)

    # Generate and save segmented image
    segmented_image = draw_segmentation_map(output.cpu())
    final_image = image_overlay(image_resized, segmented_image)
    cv2.imwrite(output_path, final_image)  # Ensure colors remain unchanged

# Compute mean IoUs
mean_iou_concrete = np.nanmean(iou_scores_concrete)
mean_iou_formwork = np.nanmean(iou_scores_formwork)
total_mean_iou = np.nanmean([mean_iou_concrete, mean_iou_formwork])

# Print Mean IoUs
print("")
print('-' * 50)
print(f"Mean IoU (Class 1): {mean_iou_concrete:.4f}")
print(f"Mean IoU (Class 2): {mean_iou_formwork:.4f}")
print(f"Total Mean IoU: {total_mean_iou:.4f}")
print('-' * 50)
print("")
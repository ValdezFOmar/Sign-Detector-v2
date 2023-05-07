import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector


# Centers the image on a white background
def center_Image(img: any, hands: list, IMG_SIZE=300, OFFSET=20):
    hand = hands[0]
    bbox_x, bbox_y, bbox_w, bbox_h = hand["bbox"]

    template = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    # Extract the image within this coordinates
    y = bbox_y - OFFSET
    h = bbox_y + bbox_h + OFFSET
    x = bbox_x - OFFSET
    w = bbox_x + bbox_w + OFFSET
    crop_img = img[y:h, x:w]

    # Height and width of the crop image
    crop_img_h = crop_img.shape[0]
    crop_img_w = crop_img.shape[1]

    # One of the sides might not be properly defined
    if crop_img_h <= 0 or crop_img_w <= 0:
        return template

    aspecRatio = crop_img_h / crop_img_w
    try:
        # If the image is too tall, adjust the width
        # and make the height = IMG_SIZE
        if aspecRatio > 1:
            scale = IMG_SIZE / crop_img_h
            new_w = math.floor(scale * crop_img_w)
            imgResized = cv2.resize(crop_img, (new_w, IMG_SIZE))
            w_gap = (IMG_SIZE - new_w) // 2
            template[:, w_gap : new_w + w_gap] = imgResized
        # If the image is too wide, adjust the height
        # and make the width = IMG_SIZE
        else:
            scale = IMG_SIZE / crop_img_w
            new_h = math.floor(scale * crop_img_h)
            imgResized = cv2.resize(crop_img, (IMG_SIZE, new_h))
            h_gap = (IMG_SIZE - new_h) // 2
            template[h_gap : new_h + h_gap, :] = imgResized
        return template  # Returns the Image resized and crop

    except ValueError:
        return template  # Returns blank image


# Creates the directories for the collected data
def make_dirs(path: str, sub_dirs: list[str]):
    for dir in sub_dirs:
        dir_path = os.path.join(path, dir)
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created")


# Moves all the extra files to a extra dir
def trim_output(path: str, max_size: int):
    last_dir_name = path.split("/")[-1]
    new_dir_path = os.path.join("data", "extra", last_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    current_files = os.listdir(path)
    current_num_files = len(current_files)

    if current_num_files <= max_size:
        return

    dif = current_num_files - max_size
    files_to_delete = current_files[0:dif]

    for file in files_to_delete:
        src = os.path.join(path, file)
        dst = os.path.join(new_dir_path, file)
        os.rename(src, dst)
    print(f"Moved extra files from {path}")
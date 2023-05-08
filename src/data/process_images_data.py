import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from data_utils import make_dirs, trim_output, center_Image


# If it detects a hand, returns processed image, otherwise returns None
def process_img(img, detector: HandDetector, IMG_SIZE=300):
    BLANK_IMAGE = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    hands, imgOutput = detector.findHands(img)
    if not hands:
        return None
    centeredImage = center_Image(imgOutput, hands, IMG_SIZE=IMG_SIZE, OFFSET=10)
    if np.array_equal(centeredImage, BLANK_IMAGE):
        return None
    return centeredImage


# Script for processing raw data:
# Get the raw images and converts them
# into images for training (kp drawn)
def main():
    num_converted_img = 0
    dict_converted_img = {}

    RAW_DATA_PATH = "data/asl_alphabet_raw"
    PROCESS_DATA_PATH = "data/asl_alphabet_processed"
    SIGNS = os.listdir(RAW_DATA_PATH)
    SIGNS.remove("none")
    SIGNS.sort()

    detector = HandDetector(maxHands=1)

    make_dirs(PROCESS_DATA_PATH, SIGNS)

    for sign in SIGNS:
        process_sign_dir = os.path.join(PROCESS_DATA_PATH, sign)
        raw_imgs_path = os.path.join(RAW_DATA_PATH, sign)

        for img_name in os.listdir(raw_imgs_path):
            img = cv2.imread(os.path.join(raw_imgs_path, img_name))
            image_save = process_img(img, detector)

            if image_save is None:
                continue
            cv2.imwrite(os.path.join(process_sign_dir, img_name), image_save)
            num_converted_img += 1

        print(f"Converted {num_converted_img} images for sign '{sign}'")
        dict_converted_img[sign] = num_converted_img
        num_converted_img = 0

    # Number of converted images per sign
    print(dict_converted_img)

    # Moves the extra images, so all dirs have the same ammount
    minimum = min(list(dict_converted_img.values()))
    print(minimum)
    for sign in SIGNS:
        trim_output(os.path.join(PROCESS_DATA_PATH, sign), minimum)


if __name__ == "__main__":
    main()

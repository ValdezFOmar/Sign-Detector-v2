import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from data_collection import center_Image, IMG_SIZE


# TODO. Refactor this garbage code and add better way to check if the image is blank
# If it detects a hand, returns the image, otherwise returns None
def convert_img(img, detector: HandDetector):
    blank_image = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    hands, imgOutput = detector.findHands(img)
    try:
        if not hands:
            return None
        centeredImage = center_Image(imgOutput, hands)
        if np.array_equal(centeredImage, blank_image):
            return None
        return centeredImage
    except ValueError:
        return None


# Get the images from the raw data and converts them into
# data for training
def main():
    num_processed_img = 0

    RAW_DATA_PATH = "data/asl_alphabet_raw"
    PROCESS_DATA_PATH = "data/asl_alphabet_processed"
    SIGNS = os.listdir(RAW_DATA_PATH)
    SIGNS.remove("none")
    SIGNS.sort()

    detector = HandDetector(maxHands=1)

    for sign in SIGNS:
        new_sign_dir = os.path.join(PROCESS_DATA_PATH, sign)
        try:
            os.mkdir(new_sign_dir)
        except FileExistsError:
            pass

        sign_path = os.path.join(RAW_DATA_PATH, sign)
        for img_name in os.listdir(sign_path):
            img = cv2.imread(os.path.join(sign_path, img_name))
            image_save = convert_img(img, detector)
            if image_save is not None:
                cv2.imwrite(os.path.join(new_sign_dir, img_name), image_save)
                num_processed_img += 1
            if not num_processed_img < 100:
                break
        print(f"Prossed {num_processed_img} images for sign '{sign}'")
        num_processed_img = 0


if __name__ == "__main__":
    main()

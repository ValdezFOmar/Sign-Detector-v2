import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from data_collection import center_Image, make_dirs, IMG_SIZE


BLANK_IMAGE = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255


# If it detects a hand, returns the image, otherwise returns None
def process_img(img, detector: HandDetector):
    hands, imgOutput = detector.findHands(img)
    if not hands:
        return None
    centeredImage = center_Image(imgOutput, hands)
    if np.array_equal(centeredImage, BLANK_IMAGE):
        return None
    return centeredImage


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


# Get the images from the raw data and
# converts them into data for training
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

    # Moved the extra images, so all dirs have the same ammount
    minimum = min(list(dict_converted_img.values()))
    print(minimum)
    for sign in SIGNS:
        trim_output(os.path.join(PROCESS_DATA_PATH, sign), minimum)

if __name__ == "__main__":
    main()

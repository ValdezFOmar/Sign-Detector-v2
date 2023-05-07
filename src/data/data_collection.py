import cv2
import numpy as np
import math
import time
import os
from cvzone.HandTrackingModule import HandDetector

IMG_SIZE = 300

# Centers the image on a white background
def center_Image(img: any, hands: list):
    OFFSET = 20
    hand = hands[0]
    bbox_x, bbox_y, bbox_w, bbox_h = hand["bbox"]

    template = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    
    # Extract the image within this coordinates
    y = bbox_y - OFFSET
    h = bbox_y + bbox_h + OFFSET
    x = bbox_x - OFFSET
    w = bbox_x + bbox_w + OFFSET
    crop_img = img[ y : h, x : w ]
    
    # Height and width of the crop image
    crop_img_h = crop_img.shape[0]
    crop_img_w = crop_img.shape[1]
    
    # One of the sides might not be properly defined
    if crop_img_h <= 0 or crop_img_w <= 0:
        return template

    aspecRatio = crop_img_h / crop_img_w
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
        template[h_gap : new_h + h_gap,:] = imgResized
    
    return template


# Creates the directories for the collected data
def make_dirs(path: str, sub_dirs: list[str]):
    for dir in sub_dirs:
        dir_path = os.path.join(path, dir)
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created")


# Opens camera feed and process the hand display
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    counter = 0
    current_sign = 0
    
    TOTAL_IMAGES = 200
    DATA_PATH = os.path.join("data","test")
    SIGNS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M','N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
        'X', 'Y', 'Z',
    ]
    
    make_dirs(DATA_PATH, SIGNS)

    while cap.isOpened():
        _, img = cap.read(0)
        hands, img = detector.findHands(img)

        if hands:
            centeredImage = center_Image(img, hands)
            cv2.imshow("Centered image", centeredImage)

        fliped_img = cv2.flip(img, 1)
        display_text = f"Saved image: {SIGNS[current_sign]} - {counter + 1}/{TOTAL_IMAGES}"
        cv2.putText(
            img=fliped_img,
            text=display_text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            color=(255, 255, 255),
            fontScale=1,
            thickness=4,                      
        )
        
        cv2.imshow("Processing frames", fliped_img)

        key = cv2.waitKey(1)
        # Exit when 'q' is pressed or
        # when all the signs have been iterated
        if key == ord("q") or current_sign == len(SIGNS):
            break
        if key == ord("s"):  # Save image to data dir
            counter += 1
            path_to_save = os.path.join(
                DATA_PATH, SIGNS[current_sign],
                f"{SIGNS[current_sign]}_{time.time()}.jpg"
            )
            cv2.imwrite(path_to_save, centeredImage)
            print(display_text)
            
            if not counter < TOTAL_IMAGES:
                counter = 0
                current_sign += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

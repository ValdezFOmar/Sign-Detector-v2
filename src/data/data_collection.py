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
    x, y, w, h = hand["bbox"]

    white_bg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    imgWhite = white_bg

    imgCrop = img[y - OFFSET : y + h + OFFSET, x - OFFSET : x + w + OFFSET]

    aspecRatio = h / w
    try:
        if aspecRatio > 1:
            k = IMG_SIZE / h
            wCalc = math.ceil(k * w)
            imgResized = cv2.resize(imgCrop, (wCalc, IMG_SIZE))
            wGap = math.ceil((IMG_SIZE - wCalc) / 2)
            imgWhite[:, wGap : wCalc + wGap] = imgResized
        else:
            k = IMG_SIZE / w
            hCalc = math.ceil(k * h)
            imgResized = cv2.resize(imgCrop, (IMG_SIZE, hCalc))
            hGap = math.ceil((IMG_SIZE - hCalc) / 2)
            imgWhite[hGap : hCalc + hGap, :] = imgResized
    except cv2.error:  # The hand might not be properly resized
        imgWhite = white_bg  # Set it to a white background

    return imgWhite


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
    SIGNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
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
        
        cv2.imshow("Image", fliped_img)

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

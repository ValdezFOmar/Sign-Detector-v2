import cv2
import time
import os
from data_utils import center_Image, make_dirs
from cvzone.HandTrackingModule import HandDetector


# Script for data collection:
# Opens camera feed and saves the hand displayed
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

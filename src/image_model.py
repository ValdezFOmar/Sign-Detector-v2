import cv2
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
from data.data_utils import center_Image


# Opens camera feed and predicts the hand sign
def main():
    KERAS_MODEL_PATH = "models/MX_alphabet_model/keras_model.h5"
    LABELS_PATH = "models/MX_alphabet_model/labels.txt"
    lines = open(LABELS_PATH, "r").readlines()
    LABELS = [line[-2] for line in lines]
    print(LABELS)

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier(KERAS_MODEL_PATH, LABELS_PATH)

    while cap.isOpened():
        _, img = cap.read(0)
        imgOutput = cv2.flip(img.copy(), 1)  # Image to display (without keypoints)

        hands, img = detector.findHands(img)

        # BG for displaying the prediction
        cv2.rectangle(
            img=imgOutput,
            pt1=(0, 0),
            pt2=(imgOutput.shape[1], 40),
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )

        # Logic for prediction
        if hands:
            centeredImage = center_Image(img, hands)
            results, index = classifier.getPrediction(centeredImage)
            confidense = results[index]
            label = LABELS[index]
            
            prediction = f"'{label}' [{int(confidense * 100)}%]"
            display_text = prediction if confidense > 0.5 else "No se reconoce"
        else:
            display_text = "No se detectan manos"

        # Writes prediction to frame
        cv2.putText(
            img=imgOutput,
            text=display_text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            color=(0, 0, 0),
            fontScale=1,
            thickness=4,
        )

        cv2.imshow("Image", imgOutput)

        key = cv2.waitKey(1)
        if key == ord("q"):  # Exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
from fer import FER

def detect_emotion():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            print("Detected Emotion:", top_emotion)

            cv2.putText(frame, f"Emotion: {top_emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


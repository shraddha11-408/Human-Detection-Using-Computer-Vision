import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('People.mp4')

while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect bodies with confidence scores
    bodies = face_cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5, outputRejectLevels=True
    )

    # Extract rectangles and confidence scores
    rectangles, reject_levels, level_weights = bodies

    for (x, y, w, h), confidence in zip(rectangles, level_weights):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

        # Display confidence score
        cv2.putText(img, f'Confidence: {confidence:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

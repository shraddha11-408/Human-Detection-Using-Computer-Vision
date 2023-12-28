# Import OpenCV library- TASK 1
import cv2

# Open the laptop camera
cam = cv2.VideoCapture(0)

# Load the pre-trained HOG-based pedestrian detector- TASK 2
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # Read the frame from the camera
    ret, frame = cam.read()

    # Convert the frame to grayscale- TASK 3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to a smaller size for faster processing
    gray = cv2.resize(gray, (640, 480))

    # Detect pedestrians in the grayscale frame
    boxes, weights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # Draw bounding boxes around the detected pedestrians
    for (x, y, w, h), confidence in zip(boxes, weights):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f'Confidence: {round(confidence, 2)}'
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes- TASK 4
    cv2.imshow('Pedestrian Detection', frame)
    
    # Retrieve the bounding boxes and confidence scores and display the number of humans detected- TASK 5
    count = len(boxes)
    print(f"Number of Humans Detected: {count}")

    # Exit if the 'ESC' is pressed- TASK 6
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
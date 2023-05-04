import cv2
import mediapipe as mp

# Initialize the Mediapipe hand detection model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Flip the webcam feed horizontally
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the Mediapipe hand detection module
hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally
    frame = cv2.flip(frame, flipCode=1)

    # Convert the frame to RGB and pass it to the Mediapipe hand detection model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image)


    # Draw the hand landmarks on the output image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the output image
    cv2.imshow('Hand Detection', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()



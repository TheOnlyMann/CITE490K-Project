import cv2
import mediapipe as mp

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video file
cap = cv2.VideoCapture(0)

# Loop over each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the pose model on the frame
    results = pose.process(image)

    # Draw the pose skeleton on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0)),
                              mp_drawing.DrawingSpec(color=(0, 255, 0)))

    # Show the frame
    cv2.imshow("Video", frame)

    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
cap.release()

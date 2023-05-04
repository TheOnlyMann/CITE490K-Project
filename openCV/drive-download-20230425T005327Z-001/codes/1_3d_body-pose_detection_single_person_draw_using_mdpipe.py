import cv2
import mediapipe as mp

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load sample image
image = cv2.imread("images/test.jpg")

# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Run the model on the image
results = pose.process(image)

# Draw the pose skeleton on the image
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                          mp_drawing.DrawingSpec(color=(255, 0, 0)),
                          mp_drawing.DrawingSpec(color=(0, 255, 0)))

# Convert the image back to BGR
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Show the image
cv2.imshow("Image", image)
cv2.waitKey(0)

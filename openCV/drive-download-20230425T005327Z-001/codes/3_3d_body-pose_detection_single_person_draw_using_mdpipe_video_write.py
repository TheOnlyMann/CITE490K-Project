import cv2
import mediapipe as mp

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video file
cap = cv2.VideoCapture("vids/vid1.mp4")

# Get the video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 25.0, (width, height))

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
    # Resize the frame
    new_h = height//2
    new_w = width//2
    resized_frame = cv2.resize(frame,(new_w,new_h))

    # Show the frame
    cv2.imshow("Video", resized_frame)

    # Write the frame to the output video file
    out.write(frame)

    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
cap.release()
out.release()




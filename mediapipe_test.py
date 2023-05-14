import cv2
import mediapipe as mp
import pandas as pd

# Video file path
video_path = "C:/Users/ahn hyeontae/Videos/4K Video Downloader/Don't Say Goodbye - Line Dance (Demo & Walk Through).mp4"
pose_det_conf = 0.5
pose_trk_conf = 0.5

# Body keypoints name
body_keypoints_name = {
    0: "NOSE",
    1: "NECK",
    2: "RIGHT_SHOULDER",
    3: "RIGHT_ELBOW",
    4: "RIGHT_WRIST",
    5: "LEFT_SHOULDER",
    6: "LEFT_ELBOW",
    7: "LEFT_WRIST",
    8: "RIGHT_HIP",
    9: "RIGHT_KNEE",
    10: "RIGHT_ANKLE",
    11: "LEFT_HIP",
    12: "LEFT_KNEE",
    13: "LEFT_ANKLE",
    14: "RIGHT_EYE",
    15: "LEFT_EYE",
    16: "RIGHT_EAR",
    17: "LEFT_EAR"
}

# Step 1: Set up the environment
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Step 2: Initialize the video capture
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Step 3: Set up MediaPipe Pose detection
with mp_pose.Pose(min_detection_confidence=pose_det_conf, min_tracking_confidence=pose_trk_conf) as pose:
    pose_data = []

    # Step 4: Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pass RGB frame to MediaPipe Pose detection
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Retrieve pose landmarks
            landmarks = results.pose_landmarks.landmark

            # Store the X, Y, and Z coordinates
            landmark_data = [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]

            pose_data.append(landmark_data)

        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0)),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))

        # Display the resulting frame
        cv2.imshow('Video Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Create a pandas DataFrame from the pose data
    df = pd.DataFrame(pose_data)

    # Save the DataFrame to a CSV file
    df.to_csv('pose_landmarks.csv', index=False)

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()






'''
img = cv2.imread("images/text.jpg")
cap = cv2.Videocapture("")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
pose = mp.pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

result = pose.process(img)
print(result.pose_landmarks)
mp_drawing = mp.solutions.drawing_utils
'''
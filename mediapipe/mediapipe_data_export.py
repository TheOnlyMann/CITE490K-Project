import os
import cv2
import mediapipe as mp
import pandas as pd

# Step 1: Set up the environment
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Step 2: Specify the directory containing the videos
videos_directory = "C:/Users/ahn hyeontae/OneDrive - postech.ac.kr/POSTECH 자료/2023-01/CITE490K 패브리케이션/video"

# Step 3: Specify the subdirectory for CSV files
csv_directory = "C:/Users/ahn hyeontae/Documents/GitHub/CITE490K-Project/csv_extraction"

# Specify the maximum allowed dimensions for the frame
max_width = 720
max_height = 480

mp_landmarks = ["nose", 
                "left eye(inner)", "left eye","left eye(outer)",
                "right eye(inner)""right eye","right eye(outer)"
                "left ear",
                "right ear",
                "mouth(left)",
                "mouth(right)",
                "left shoulder",
                "right shoulder",
                "left elbow",
                "right elbow",
                "left wrist",
                "right wrist",
                "left pinky",
                "right pinky",
                "left index",
                "right index",
                "left thumb",
                "right thumb",
                "left hip",
                "right hip",
                "left knee",
                "right knee",
                "left ankle",
                "right ankle",
                "left heel",
                "right heel",
                "left foot index",
                "right foot index"]

# Step 4: Set up MediaPipe Pose detection
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Step 5: Iterate over the video files in the directory
    for filename in os.listdir(videos_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(videos_directory, filename)

            # Step 6: Initialize video capture
            cap = cv2.VideoCapture(video_path)

            # Step 7: Initialize pose data list
            pose_data = []

            # Step 8: Process video frames
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
                    landmark_data = {"frame_data":cap.get(cv2.CAP_PROP_POS_FRAMES)}+{mp_landmarks[i]:[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))}

                    pose_data.append(landmark_data)

                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0)),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))

                # Display the resulting frame
                # Get the dimensions of the frame
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                aspect_ratio = min(max_width / frame_width, max_height / frame_height, 1)
                new_width = int(frame_width * aspect_ratio)
                new_height = int(frame_height * aspect_ratio)
                new_frame = cv2.resize(frame, (new_width, new_height))
                cv2.imshow('Video Pose Detection', new_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Step 9: Create a pandas DataFrame from the pose data
            df = pd.DataFrame(pose_data)

            # Step 10: Create the output CSV file path in the specified subdirectory
            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(csv_directory, csv_filename)

            # Step 11: Save the DataFrame to a CSV file
            df.to_csv(csv_path, index=False)

            # Step 12: Release the video capture
            cap.release()

    # Step 13: Close all windows
    cv2.destroyAllWindows()
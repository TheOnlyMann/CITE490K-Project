import datetime
import cv2
import mediapipe as mp
import pandas as pd

# Mediapipe Pose Model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#initialize pose model
pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

cap = cv2.VideoCapture("video/Don't Say Goodbye - Line Dance (Demo & Walk Through).mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

def export_as_csv(obj): # Save per frame

    # Body ID, Date, and Time columns
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S.%f")

    detection_results = {"Body_ID" : obj.id, "Date" : date_string, "Time" : time_string}

    # Keypoint columns
    for i , keypnt in enumerate(obj.keypoint):
        keypnt_str = ','.join(str(x) for x in keypnt)
        detection_results[str(i)+"_"+body_keypoints_name[i]] = keypnt_str

    # Convert the detection results to a DataFrame
    df = pd.DataFrame(detection_results,index=[0])
        
    # Append the DataFrame to the CSV file
    file_name = 'detection_results.csv'
    with open(file_name, 'a', newline='') as f:
        df.to_csv(f, header=f.tell()==0, index=False, mode='a', lineterminator="")               
        



#frame looping
while True:
    #1 frame 
    ret, frame = cap.read()
    if not ret:
        break#end if no next frame
    #convert color BGR->RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #result
    result = pose.process(img)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,0,0)),mp_drawing.DrawingSpec(color=(0,255,0)))

    new_h=height//2
    new_w=width//2
    resized_frame = cv2.resize(frame,(new_w,new_h))
    cv2.imshow('Video', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#testing if editing does work
#working?
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
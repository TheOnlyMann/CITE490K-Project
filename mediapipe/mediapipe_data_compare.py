import os
import cv2
import mediapipe as mp
import pandas as pd
from difflib import SequenceMatcher

# Step 3: Specify the subdirectory for CSV files
csv_directory = "C:/Users/ahn hyeontae/Documents/GitHub/CITE490K-Project/csv_extraction"
csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]

mp_landmarks = ["nose", 
                "left eye(inner)", "left eye","left eye(outer)",
                "right eye(inner)","right eye","right eye(outer)",
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

#read 2 videos, one for front and other for back
file1 = input("Name of the front video:")
file2 = input("Name of the back video:")
sim_f1 = [SequenceMatcher(None, file1, csv_file).ratio() for csv_file in csv_files]
sim_f2 = [SequenceMatcher(None, file2, csv_file).ratio() for csv_file in csv_files]
msi_f1 = sim_f1.index(max(sim_f1))
msi_f2 = sim_f2.index(max(sim_f2))
msc_f1 = csv_files[msi_f1]
msc_f2 = csv_files[msi_f2]
csv_path_f1 = os.path.join(csv_directory, msc_f1)
csv_path_f2 = os.path.join(csv_directory, msc_f2)
df_f1=pd.read_csv(csv_path_f1)
df_f2=pd.read_csv(csv_path_f2)


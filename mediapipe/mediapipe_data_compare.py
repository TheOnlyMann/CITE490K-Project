import os
import cv2
import pandas as pd
from dtw import dtw
import numpy as np
from difflib import SequenceMatcher

# Step 1: set variables
max_frame = 150
max_sim_dist = 35
min_error_dist = 0.0003
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
print(f"First:{msc_f1}, Second:{msc_f2}")
csv_path_f1 = os.path.join(csv_directory, msc_f1)
csv_path_f2 = os.path.join(csv_directory, msc_f2)
df_f1=pd.read_csv(csv_path_f1)
df_f2=pd.read_csv(csv_path_f2)


# Define the distance function for DTW
def distance(x, y):
    x_coords = np.array([float(coord) for coord in x.split('/')])
    y_coords = np.array([float(coord) for coord in y.split('/')])
    return np.linalg.norm(x_coords - y_coords)

# Perform DTW between the first and second DataFrames
dist, cost, acc, path = dtw(df_f1.iloc[:, 1:], df_f2.iloc[:, 1:], dist=distance)

# Calculate the number of overlapping frames
overlap_frames = len(df_f1) - path[-1][-1] + 1


# Calculate average displacement for each coordinate (X, Y, Z)
avg_displacement = df_f1.iloc[-overlap_frames:, 1:].apply(lambda x: pd.to_numeric(x.str.split('/').str[0])).mean(), \
                  df_f1.iloc[-overlap_frames:, 1:].apply(lambda x: pd.to_numeric(x.str.split('/').str[1])).mean(), \
                  df_f1.iloc[-overlap_frames:, 1:].apply(lambda x: pd.to_numeric(x.str.split('/').str[2])).mean()

# Apply displacement to the second DataFrame
df_f2_corrected = df_f2.copy()
df_f2_corrected.iloc[:overlap_frames, 1:] = (df_f2_corrected.iloc[:overlap_frames, 1:].apply(lambda x: pd.to_numeric(x.str.split('/').str[0])) + avg_displacement[0]).astype(str) + '/' + \
                                            (df_f2_corrected.iloc[:overlap_frames, 1:].apply(lambda x: pd.to_numeric(x.str.split('/').str[1])) + avg_displacement[1]).astype(str) + '/' + \
                                            (df_f2_corrected.iloc[:overlap_frames, 1:].apply(lambda x: pd.to_numeric(x.str.split('/').str[2])) + avg_displacement[2]).astype(str)

# Combine the DataFrames
combined_df = pd.concat([df_f1.head(len(df_f1) - overlap_frames), df_f2_corrected, df_f2.tail(len(df_f2) - overlap_frames)])


# Step 8: Save the combined DataFrame as a CSV file
filefinal = input("Save as:")
csv_filename = filefinal + ".csv"
csv_path = os.path.join(csv_directory, csv_filename)
combined_df.to_csv(csv_path, index=False)
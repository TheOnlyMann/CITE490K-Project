import os
import cv2
import mediapipe as mp
import pandas as pd
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

def df_chk_dif(a,b):#gets two dataframe in, returns their average displacement and how to correct the latter one
    if len(a)!= len(b): return -1
    tot_sum=0
    disp=[0,0,0]
    for i in range(len(a)):
        aic=a.iloc[i,1].split('/')
        bic=b.iloc[i,1].split('/')
        for j in list(range(8,len(a.columns))):
            aij=a.iloc[i,j].split('/')
            bij=b.iloc[i,j].split('/')
            euc_sum = 0
            for k in range(3):
                euc_sum += abs((float(aij[k])-float(aic[k]))-(float(bij[k])-float(bic[k])))
            if euc_sum > min_error_dist:#only add big changes
                tot_sum+=euc_sum
        for k in range(3):
            disp[k]+=float(aic[k])/len(a)-float(bic[k])/len(a)
    return tot_sum, disp
        
def df_pp(a,c):
    for i in range(len(a)):
        for j in list(range(2,len(a.columns))):
            aij=a.iloc[i,j].split('/')
            a.loc[i][j]=f"{float(aij[0])+c[0]}/{float(aij[1])+c[1]}/{float(aij[2])+c[2]}"

def df_avg(a,b):
    c=a.copy()
    for i in range(len(a)):
        for j in list(range(2,len(a.columns))):
            aij=a.iloc[i,j].split('/')
            bij=b.iloc[i,j].split('/')
            c.loc[i][j]=f"{(float(aij[0])+float(bij[0]))/2}/{(float(aij[1])+float(bij[1]))/2}/{(float(aij[2])+float(bij[2]))/2}"
    return c

min_state_dist = max_sim_dist
min_state_disp =[0,0,0]
min_state_over = 0
for i in range(0,min(max_frame,len(df_f1),len(df_f2))):#do a for from 1 to maximum frames, which is about 5 seconds.
    dt_f1 = df_f1.tail(i)
    dt_f2 = df_f2.head(i)
    #if i == 0 :#check for "zero"
    state_dist, state_disp = df_chk_dif(dt_f1,dt_f2)
    if not min_state_dist or i == 1 or min_state_dist>state_dist:
        min_state_dist=state_dist
        min_state_disp=state_disp
        min_state_over=i
    print(f"testing {i} frames overlapped, with accruacy {state_dist} and displacement {state_disp}")
assert max_sim_dist>min_state_dist
print(f"using {min_state_over} frames overlapped, with accruacy {min_state_dist} and displacement {min_state_disp}")
df_pp(df_f2,min_state_disp)
result=pd.concat([df_f1.head(len(df_f1)-min_state_over),df_avg(df_f1.tail(min_state_over),df_f2.head(min_state_over)),df_f1.tail(len(df_f2)-min_state_over)],axis=0)


filefinal = input("Save as:")
csv_filename = filefinal + ".csv"
csv_path = os.path.join(csv_directory, csv_filename)
result.to_csv(csv_path, index=False)
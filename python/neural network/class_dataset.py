#Categorical databases
import pandas as pd
import numpy as np
import os
import shutil
import re
from sklearn.model_selection import train_test_split

image_dir = "/mnt/d/prosessed_imag"
output_dir = "dataset"
train_ratio = 0.8
file_path = '/home/liushuai/seed/seedprosessing/sorted_output.xlsx'



all_sheets = pd.read_excel(file_path,sheet_name=None,header=None)

ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],\
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']\
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = np.concatenate((ori_data[:, 1:7], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)
labels = data[:,6]

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))

if len(image_files) != len(labels):
    raise ValueError(f"image_files_len:{len(image_files)}labels_len{len(labels)}")

data_img = list(zip(image_files,labels))

train_data,test_data = train_test_split(data_img,train_size=train_ratio,stratify=labels,random_state=42)

def copy_files(data, subset):
    for filename, label in data:
        src = os.path.join(image_dir, filename)
        dst_dir = os.path.join(output_dir, subset, f"class{label}")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        shutil.copy2(src, dst)

copy_files(train_data, "train")
copy_files(test_data, "test")

print("finished")


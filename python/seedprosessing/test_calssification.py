import cv2
import re
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sympy.physics.units.definitions.dimension_definitions import information
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import shap
#user-defined
n_clusters = 5
random_state = 42
image_dir = "/mnt/d/prosessed_imag"
feature_list = []
filenames = []
image_files = []
excel_file_path = 'sorted_output.xlsx'
weigth_idx = 0

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return binary

def extract_shape_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    extent = float(area)/(w*h)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    information_else = np.array([area,perimeter,aspect_ratio,extent,solidity])
    information_end = np.concatenate([information_else,hu_moments])
    return information_end
    # return hu_moments


#import excel data
all_sheets = pd.read_excel(excel_file_path,sheet_name=None,header=None)
ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],\
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']\
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = np.concatenate((ori_data[:, 1:8], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)
data_vitality = data[:,7]
# weigth = data[:,2]
# image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
# image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))
# for filename in image_files:
#     image_path = os.path.join(image_dir, filename)
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = (img > 0).astype(np.uint8) * 255
#     features = extract_shape_features(img)
#     weight_inf = np.array([weigth[weigth_idx],weigth[weigth_idx]/features[0]*1000000])
#     features = np.concatenate([weight_inf,features])
#     weigth_idx += 1
#     feature_list.append(features)
#     filenames.append(filename)
# X = np.array(feature_list)
X = data[:,0:6]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,data_vitality,test_size = 0.2,random_state= 42)

#LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train,Y_train)
y_lr = model_lr.predict(X_test)
#RandomForestClassifier()
model_fc = RandomForestClassifier()
model_fc.fit(X_train,Y_train)
y_fc = model_fc.predict(X_test)
#XGBOOST
model_xg = XGBClassifier()
model_xg.fit(X_train,Y_train)
y_xg = model_xg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_lr))
print("Random Forest Accuracy:", accuracy_score(Y_test, y_fc))
print("XGBoost Accuracy:", accuracy_score(Y_test, y_xg))
y_proba_lr = model_lr.predict_proba(X_test)[:, 1]
print("Logistic Regression AUC:", roc_auc_score(Y_test, y_proba_lr))
feature_names = ["p","class","original weigth","day1 weigth","day2 weigth","day3 weigth"]
X_df = pd.DataFrame(X_scaled, columns=feature_names)
coef_df = pd.DataFrame({
    "feature":X_df.columns,
    "coefficient":model_lr.coef_[0]
})
coef_df = coef_df.sort_values(by="coefficient",key = abs,ascending = False)
print("Logistic Regression\n",coef_df)

importance_rf = pd.DataFrame({
    "feature":X_df.columns,
    "importance":model_fc.feature_importances_
}).sort_values("importance",ascending = False)
print("Random Forest Importance:\n", importance_rf)

importance_xg = pd.DataFrame({
    "feature":X_df.columns,
    "importance":model_xg.feature_importances_
}).sort_values("importance",ascending = False)
print("XGBoost Importance:\n", importance_xg)

explainer = shap.Explainer(model_xg)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, feature_names=X_df.columns)

plt.figure(figsize=(10, 6))
plt.barh(importance_rf["feature"], importance_rf["importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()
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
from sympy import false
from sympy.physics.units.definitions.dimension_definitions import information
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import shap
from sklearn.preprocessing import PolynomialFeatures
import itertools
from sklearn.linear_model import LogisticRegressionCV
#user-defined
n_clusters = 5
random_state = 42
image_dir = "/mnt/d/prosessed_imag"
feature_list = []
filenames = []
image_files = []
excel_file_path = 'sorted_output.xlsx'
weigth_idx = 0

def scale_with_columns(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

def add_symmetric_features(X):
    feature_names = X.columns
    new_features = []
    new_feature_names = []

    for i, j in itertools.combinations_with_replacement(feature_names, 2):
        col_i, col_j = X[i], X[j]
        if i != j:
            new_features.append(col_i + col_j)
            new_feature_names.append(f"{i}_plus_{j}")

            new_features.append((col_i - col_j).abs())
            new_feature_names.append(f"{i}_minusabs_{j}")



    new_feature_df = pd.concat(new_features, axis=1)
    new_feature_df.columns = new_feature_names

    return new_feature_df

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
weigth = data[:,2]
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))
for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img > 0).astype(np.uint8) * 255
    features = extract_shape_features(img)
    weight_inf = np.array([weigth[weigth_idx],weigth[weigth_idx]/features[0]*1000000])
    features = np.concatenate([weight_inf,features])
    weigth_idx += 1
    feature_list.append(features)
    filenames.append(filename)

X = np.array(feature_list)
feature_names = ["weigth","weigth_area","area","perimeter","aspect_ratio","extent","solidity","Hu1","Hu2","Hu3","Hu4","Hu5","Hu6","Hu7"]
X = pd.DataFrame(X, columns=feature_names)
new_feature = add_symmetric_features(X)
poly = PolynomialFeatures(degree = 2,interaction_only=False,include_bias=False)
X = pd.DataFrame(poly.fit_transform(X),columns=poly.get_feature_names_out())
X_df = pd.concat([X, new_feature], axis=1)
X_df = scale_with_columns(X_df)

X_train,X_test,Y_train,Y_test = train_test_split(X_df,data_vitality,test_size = 0.2,random_state= 42)


model_l1 = LogisticRegressionCV(
    penalty="l1", solver="liblinear", cv=5, scoring="roc_auc", max_iter=10000
)
model_l1.fit(X_train, Y_train)
y_lr = model_l1.predict(X_test)

# XGBoost
model_xgb = XGBClassifier(
    objective="binary:logistic", max_depth=3, reg_lambda=1.0, reg_alpha=0.5,
    subsample=0.8, eval_metric="auc"
)
model_xgb.fit(X_train, Y_train)
y_xg = model_xgb.predict(X_test)
#RandomForestClassifier()
model_fc = RandomForestClassifier()
model_fc.fit(X_train,Y_train)
y_fc = model_fc.predict(X_test)


coef_df = pd.DataFrame({
    "feature": X_df.columns,
    "coef_l1": model_l1.coef_[0]
}).sort_values(by="coef_l1", key=abs, ascending=False)

# --- XGBoost ---
importance_xgb = pd.DataFrame({
    "feature": X_df.columns,
    "importance": model_xgb.feature_importances_
}).sort_values("importance", ascending=False)
importance_rf = pd.DataFrame({
    "feature":X_df.columns,
    "importance":model_fc.feature_importances_
}).sort_values("importance",ascending = False)


print("Logistic Regression\n", coef_df)
print("XGBoost Importance:\n", importance_xgb)
print("Random Forest Importance:\n", importance_rf)

explainer = shap.Explainer(model_xgb)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, feature_names=X_df.columns)



print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_lr))
print("XGBoost Accuracy:", accuracy_score(Y_test, y_xg))
print("Random Forest Accuracy:", accuracy_score(Y_test, y_fc))








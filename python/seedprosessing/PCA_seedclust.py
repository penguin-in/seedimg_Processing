#This code is used for PCA dimension reduction and then clustering code
import cv2
import re
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sympy.physics.units.definitions.dimension_definitions import information
#user-defined
n_clusters = 5
random_state = 42


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



image_dir = "/mnt/d/prosessed_imag"
feature_list = []
filenames = []
image_files = []
excel_file_path = 'sorted_output.xlsx'
#import excel data
all_sheets = pd.read_excel(excel_file_path,sheet_name=None,header=None)
ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],\
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']\
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = np.concatenate((ori_data[:, 1:8], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)
data_vatlity = data[:,7]
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))
for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img > 0).astype(np.uint8) * 255
    features = extract_shape_features(img)
    feature_list.append(features)
    filenames.append(filename)

X = np.array(feature_list)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
labels = kmeans.fit_predict(X_pca)



plt.figure(figsize=(0.8*n_clusters*n_clusters, 0.4*n_clusters*(n_clusters+1)))
gs = gridspec.GridSpec(n_clusters, n_clusters+1)
ax1 = plt.subplot(gs[:,0:n_clusters])
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.6)
plt.title('clust of shape(PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
cmap = scatter.cmap
norm = scatter.norm
for i in range(n_clusters):
    shape = plt.subplot(gs[i,n_clusters])

    clust_idx = np.where(labels == i)[0]
    vatlity_num = sum(data_vatlity[clust_idx])
    clust_file = [filenames[idx] for idx in clust_idx]
    if clust_file:
        img_path = os.path.join(image_dir,clust_file[0])
        img_show = cv2.imread(img_path)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
        img_show = (img_show > 0).astype(np.uint8) * 255

        color_rgb = np.array(cmap(norm(i))[:3])*255
        color_rgb = color_rgb.astype(np.uint8)
        color_img = np.ones((img_show.shape[0],img_show.shape[1],3),dtype=np.uint8)*255
        mask = img_show == 255
        for c in range(3):
            color_img[:,:,c][mask] = color_rgb[c]

        shape.imshow(color_img)
        shape.axis('off')
        color = cmap(norm(i))
        shape.set_title(f'n={len(clust_file)},p={(vatlity_num / len(clust_file)*100):.2f}%', fontsize=14,pad=10)
        # shape.set_title(f'n={len(clust_file)},p={(vatlity_num/len(clust_file)):.2f}',color = color,fontsize = 14,pad = 10)
    else:
        shape.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace = 0.2,hspace = 0.2)
plt.savefig("PCA_hu7feature.png", dpi=600, bbox_inches='tight')
plt.show()
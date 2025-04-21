import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


model = models.alexnet(pretrained=True)
model.eval()


img_path = '/home/liushuai/seed/neural network/dataset/train/high_vitality/1.png'  # 替换为你的图片路径
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)


with torch.no_grad():
    x = input_tensor
    for i in range(12):
        x = model.features[i](x)
    feature_map = x.squeeze().cpu().numpy()


def show_feature_maps(fmap, num_cols=8):
    num_channels = fmap.shape[0]
    num_rows = int(np.ceil(num_channels / num_cols))
    plt.figure(figsize=(num_rows * 1.5, num_cols * 1.5))
    for i in range(num_channels):
        row = i % num_rows
        col = i // num_rows
        plt.subplot(num_cols, num_rows, i + 1)
        plt.imshow(fmap[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle('Feature Maps from ReLU5 (features[11])')
    plt.tight_layout()
    plt.show()

show_feature_maps(feature_map)


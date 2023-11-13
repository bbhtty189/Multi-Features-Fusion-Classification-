import numpy as np
import os
import cv2
from traditional_features import *

img_path = r"E:\datasets\MTARSI\airplane-datasets-new\B-1\1-1.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
cv2.imwrite(os.path.join("./test/test.jpg"), img)

h, w, _ = img.shape
child_imgs = []
x = np.linspace(0, w, 4).astype(np.longlong)
y = np.linspace(0, h, 4).astype(np.longlong)
print(img.shape)
print(len(x), x)
print(len(y), y)
for i in range(3):
    for j in range(3):
        child_img = img[y[i]:y[i+1], x[j]:x[j+1], :]
        child_imgs.append(child_img)
        cv2.imwrite(os.path.join("./test", str(i)+ " " +str(j)+".jpg"), child_img)

features = calculate_hog(img)
all_features = calculate_child_hog(img, num=3)

all_features = all_features.reshape(-1)
print(features.shape)
print(all_features.shape)



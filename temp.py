import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import cv2
import os
import numpy as np

def load_images_from_folder(folder, folder2):
    images = []
    names = []
    masks = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(folder2,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img //= 255
            mask //= 255
            images.append(img)
            names.append(filename)
            masks.append(mask)
    return np.array(images), np.array(masks), names
def dice_cofficient(y_true, y_pred, smooth=1e-8):
    intersection = np.sum(y_pred[y_true==1])
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection ) / (union)

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred)
    return np.sum(y_true == y_pred) / y_true.size


image_dir = r'test_sample'
mask_dir = r'test_sample_mask_1'
generated_mask_dir = r'generated_mask_1'
mask, gen_mask, g_names = load_images_from_folder(mask_dir, generated_mask_dir)

dc = 0
acc = 0
for i in range(gen_mask.shape[0]):
    dc = dice_cofficient(gen_mask[i],mask[i])
    acc = accuracy(mask[i],gen_mask[i])
    print(g_names[i],dc,acc)
dc /= gen_mask.shape[0]
acc /= gen_mask.shape[0]
print(dc, acc)




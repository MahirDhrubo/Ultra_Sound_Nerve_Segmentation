import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import os

def process(path, mask = False):
    image = cv2.imread(path)
    # print(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg', image)
    image = cv2.resize(image, (256,256), interpolation =cv2.INTER_AREA)
    cv2.imwrite('resize_image.jpg', image)
    if mask:
        _, image = cv2.threshold(image,190,255,cv2.THRESH_BINARY)
    # cv2.imwrite('small_process.jpg', image)
    # print(np.std(image))
    # return np.float32(image - np.mean(image)) / np.std(image)
    return np.float32(image/255)

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transorm=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transorm
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.annotations.iloc[index,1])
        image = process(image_path)
        
        mask_name = self.annotations.iloc[index,1].split('.')[0]+'_mask.tif'
        mask_path = os.path.join(self.image_dir, mask_name)
        mask = process(mask_path, mask=True)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return (image, mask)
    
    

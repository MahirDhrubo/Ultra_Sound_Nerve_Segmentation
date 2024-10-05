import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from data_loader import process
from rsaModel import RA_Net
# from skimage.metrics import binary_dice
# from keras import backend as K

class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        #encoder blocks
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)

        #decoder blocks
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
    

    def forward(self, input):
        input = self.relu(self.conv1(input))
        input = self.relu(self.conv2(input))
        input = self.relu(self.conv3(input))
        input = self.relu(self.conv4(input))

        input = self.relu(self.deconv1(input))
        input = self.relu(self.deconv2(input))
        input = self.relu(self.deconv3(input))
        input = self.deconv4(input)

        return input
    

def dice_cofficient(y_true, y_pred, smooth=1e-8):
    intersection = np.sum(y_pred*y_true)
    # tp = 0
    # tq = 0
    # for i in range(y_pred.shape[1]):
    #     for j in range(y_pred.shape[2]):
    #         if y_pred[0][i][j] != 1 and y_pred[0][i][j] != 0:
    #             # print(y_pred[0][i])
    #             tp += 1
    #         if y_true[0][i][j] != 1 and y_true[0][i][j] != 0:
    #             # print(y_pred[0][i])
    #             tq += 1
    # print(tp, tq)
    union = np.sum(y_true**2) + np.sum(y_pred**2)
    return (2.0 * intersection ) / (union)

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred)
    return np.sum(y_true == y_pred) / y_true.size

def test(model_path, image, device='cpu'):
    model = RA_Net(n_channels=1)
    model.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
    impainted_image = None
    with torch.no_grad():
        model.eval()
        impainted_image = model(image)
    
    return impainted_image

def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            temp = process(os.path.join(folder,filename))
            temp = temp[np.newaxis, :, :]
            images.append(temp)
            names.append(filename)
    return np.array(images), names



image_dir = r'test_sample'
mask_dir = r'test_sample_mask'
generated_mask_dir = r'generated_mask_2'
image, names = load_images_from_folder(image_dir)
image_mask, _ = load_images_from_folder(mask_dir)
print(image.shape)
print(image_mask.shape)

# transform = transforms.ToTensor()
# for i in range(image.shape[0]):
#     image[i] = transform(image[i])
#     image_mask[i] = transform(image_mask[i])
image = torch.from_numpy(image)
image_mask = torch.from_numpy(image_mask)
print(image.shape)
print(image_mask.shape)

# image = torch.unsqueeze(image, dim=1)
# image_mask = torch.unsqueeze(image_mask, dim=1)

new_images = test(r'D:\Academics\4-2\CSE472(ML sessional)\project_ultrasound_nerve_segmentation\UltraSoundNerve\runhistory\4\nerve_model_path.path', image)
new_images = new_images.numpy()
dc = 0
for i in range(new_images.shape[0]):
    y_true = image_mask[i]
    dc += 1 - dice_cofficient(y_true.numpy(), new_images[i])
    print(names[i],dc)
print(dc / new_images.shape[0])
image_mask = image_mask.numpy()
print(new_images.shape)
print(image_mask.shape)
# print('Dice Coefficient: ', dc)
# 0-255 pixel range
for i in range(new_images.shape[0]):
    new_image = new_images[i]
    name = names[i]
    new_image = (new_image*255).astype(np.uint8)

    new_image = cv2.resize(new_image[0], (580,420), interpolation =cv2.INTER_AREA)

    # cv2.imwrite(os.path.join(generated_mask_dir,'resized_'+name), new_image)
    _, new_image = cv2.threshold(new_image,170,255,cv2.THRESH_BINARY)
    name = name.split('.')
    name = name[0] + '_mask.tif' 
    cv2.imwrite(os.path.join(generated_mask_dir,name), new_image)


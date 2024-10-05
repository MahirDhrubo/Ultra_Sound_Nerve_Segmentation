import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np

from data_loader import CustomDataset
from rsaModel import RA_Net

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


#hpyer paramters of the model
INIT_LR = 0.00001
BATCH_SIZE = 32
EPOCHS = 20

#75% train data
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.10
VAL_SPLIT = 0.10
# assign device and print to check gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# loading the dataset
dataset = CustomDataset(csv_file='names.csv', image_dir='train',
                         transorm=transforms.ToTensor())
train_len = int(len(dataset)*TRAIN_SPLIT)
test_len = int(len(dataset)*TEST_SPLIT)
val_len = int(len(dataset)*VAL_SPLIT)
train_set, test_set, val_set = random_split(dataset, [TRAIN_SPLIT,TEST_SPLIT,VAL_SPLIT], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

train_steps = len(train_loader.dataset) // BATCH_SIZE
test_steps = len(test_loader.dataset) // BATCH_SIZE
val_steps = len(val_loader.dataset) // BATCH_SIZE

print('Dataset Loaded')
print('Train samples : {}'.format(len(train_loader.dataset)))
print('Test samples : {}'.format(len(test_loader.dataset)))
print('Validation samples : {}'.format(len(val_loader.dataset)))


# initializing model parameters
# model = ContextEncoder().to(device)
model = RA_Net(n_channels=1).to(device)
optim = optim.Adam(model.parameters(), lr = INIT_LR)
sigmoid = nn.Sigmoid()
loss_func = nn.MSELoss()

# this 'H' will save the training history
H = {
    'train_loss' : [],
    'test_loss' : [],
    'dice_score' : []
}

def dice_cofficient(y_true, y_pred, smooth=1):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

#training the network
print('Training Starts...')
start_time = time.time()
for e in tqdm(range(EPOCHS)):
    model.train()
    train_loss = 0
    test_loss = 0
    dice_score = 0

    for i,(image, real_image) in enumerate(train_loader):
        # image send to device (cpu or gpu)
        (image,real_image) = (image.to(device), real_image.to(device))

        # forward propagation
        optim.zero_grad()
        outputs = model(image)

        # calculating loss
        loss = loss_func(outputs, real_image)

        # backward propagation
        loss.backward()
        optim.step()

        train_loss += loss

    # after each epoch calculating the test loss

    with torch.no_grad():
        model.eval()

        for i,(image,real_image) in enumerate(test_loader):
            (image,real_image) = (image.to(device), real_image.to(device))

            ds = 0
            outputs = model(image)
            test_loss += loss_func(outputs, real_image)
            for j in range(outputs.shape[0]):
                ds += dice_cofficient(outputs[j], real_image[j])
            dice_score += ds / outputs.shape[0]
    
    avg_train_loss = train_loss / train_steps
    avg_test_loss = test_loss / test_steps
    avg_dice_score = dice_score / test_steps

    H['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    H['test_loss'].append(avg_test_loss.cpu().detach().numpy())
    H['dice_score'].append(avg_dice_score)

    print('[INFO] EPOCH : {}/{}'.format(e+1, EPOCHS))
    print('Train loss : {:.6f}'.format(avg_train_loss))
    print('Test loss : {:.6f}'.format(avg_test_loss))
    print('Dice score : {:.6f}'.format(avg_dice_score))

end_time = time.time()
print('Total time taken to train model : {:.2f}s'.format(end_time - start_time))

# finally plot the train loss and validation loss

plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label = 'train loss')
plt.plot(H['test_loss'], label = 'test loss')
plt.title('Training loss and Testing loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'lower left')
plt.savefig('loss_plot.png')


plt.style.use('ggplot')
plt.figure()
plt.plot(H['dice_score'], label = 'dice score')
plt.title('Dice Coefficient for test data')
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc = 'lower left')
plt.savefig('dice_plot.png')

torch.save(model.state_dict(), 'nerve_model_path.path')

print('validation')


val_loss = 0
val_dice_score = 0
avg_dice = 0
with torch.no_grad():
        model.eval()

        for image,mask_image in val_loader:
            (image,real_image) = (image.to(device), real_image.to(device))

            ds = 0
            outputs = model(image)
            val_loss += loss_func(outputs, mask_image)
            for j in range(outputs.shape[0]):
                ds += dice_cofficient(outputs[j], real_image[j])
            val_dice_score += ds / outputs.shape[0]
        avg_dice = val_dice_score / val_steps

print('Average Dice Score : {:.6f}'.format(avg_dice))

import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision.io import read_image


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.num_features = 16
        # encoder
        self.enc1 = nn.Linear(in_features=30000, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=256)
        self.enc3 = nn.Linear(in_features=256, out_features=128)
        self.enc4 = nn.Linear(in_features=128, out_features=64)
        self.enc5 = nn.Linear(in_features=64, out_features=32)
        self.enc6 = nn.Linear(in_features=32, out_features=16)
        if self.num_features == 3:
            self.enc7 = nn.Linear(in_features=16, out_features=3)
        # decoder
        if self.num_features == 3:
            self.dec0 = nn.Linear(in_features=3, out_features=16)
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=512)
        self.dec6 = nn.Linear(in_features=512, out_features=30000)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        if self.num_features == 3:
            x = F.relu(self.enc7(x))
        return x

    def decode(self, x):
        if self.num_features == 3:
            x = F.relu(self.dec0(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Extractor:
    def __init__(self, option=None):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((100, 100)),
            transforms.ToTensor()
        ])
        new_model = Autoencoder()
        #
        if option == 'load':
            new_model.load_state_dict(torch.load("./models/pomdp_autoencoder_sz16.pth", map_location=torch.device('cpu')))
        if option == 'save':
            torch.save(new_model.state_dict(), "./models/pomdp_autoencoder_new.pth")
        # Change the device to GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = new_model.to(self.device)

       #TODO latent_space_size
        self._num_features = self.model.num_features
        self._labels = ["f" + str(x) for x in np.arange(1, self._num_features+1)]
        self._type = "ae"+str(len(self._labels))

    def feature_labels(self):
        return self._labels

    def type(self):
        return self._type

    def extract(self, image_data, r):
        img = self.transform(image_data)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        img = img.view(img.size(0), -1)
        with torch.no_grad():
            # Extract the feature from the image
            features = self.model.encode(img)

        features = features.cpu().detach().numpy()[0]
        # normalize the features
        return features#/np.sum(features)

#############################
#
# Below here is for training/testing the autoencoder
#
#############################


class AE_Data(Dataset):
    def __init__(self, img_start_idx, num_imgs, dataset_path, transform=None, target_transform=None):
        self.imgs = []
        self.transform = transform
        self.target_transform = target_transform
        for img_num in range(num_imgs):
            img_path = dataset_path+'/im'+str(img_start_idx+img_num)+'.jpg'
            self.imgs.append(read_image(img_path))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = str(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float()/255.0, label


def save_decoded_image(img, epoch, decoded):
    img = img.view(img.size(0), 3, 100, 100)
    if decoded == 1:
        save_image(img, './encoder_images/linear_ae_image{}.png'.format(epoch))
    else:
        save_image(img, './encoder_images/linear_ae_image-{}.png'.format(epoch))


def train(net, trainloader, testloader, epochs, criterion, device):
    train_loss = []
    valid_loss = []

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        running_loss = 0.0
        for data in trainloader:
            img_orig, _ = data
            img = torch.clone(img_orig)
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        validation = validate(net, testloader, device)
        valid_loss.append(validation)
        print('Epoch {} of {}, Train Loss: {:.3f}, Valid Loss: {:.3f}'.format(
            epoch + 1, epochs, loss, validation))
        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch, decoded=1)
            save_decoded_image(img_orig.cpu().data, epoch, decoded=0)
    return train_loss, valid_loss


def training(trainloader, testloader, epochs, device, save_path):
    # create & load the neural network onto the device
    net = Autoencoder()
    net.to(device)
    make_dir()
    criterion = nn.MSELoss()
    # train the network
    train_loss, valid_loss = train(net, trainloader, testloader, epochs, criterion, device)
    # save the network
    torch.save(net.state_dict(), save_path)
    # plot performance
    plt.figure()
    plt.plot(train_loss, label='training')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('deep_ae_pomdp_loss.png')
    # test the network
    validate(net, testloader, criterion, device)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def make_dir():
    image_dir = 'encoder_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def validate(net, testloader, criterion, device):
    mean_ave_error = 0
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = net(img)
        loss = criterion(outputs, img)
        mean_ave_error += loss.cpu().detach().numpy()
    return mean_ave_error/len(testloader)


def plot_encoding_comparisons(img_dir, save_path):
    import cv2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale((100, 100)),
        transforms.ToTensor()])

    device = get_device()
    net = Autoencoder()
    net.load_state_dict(torch.load(save_path))
    net.to(device)

    for idx in range(800, 1000):
        img = cv2.imread(img_dir+'/im'+str(idx)+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_title('original')
        ax1.imshow(img)

        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        img = img.view(img.size(0), -1)

        outputs = net(img)

        enc = net.encode(img).cpu().data

        outputs = outputs.view(1, 3, 100, 100).cpu().data
        outputs = torch.squeeze(outputs)
        outputs = outputs.permute(1,2,0)
        ax2.set_title('decoded')
        ax2.imshow(outputs.numpy())
        fig.suptitle("{}: {}".format(idx, enc))
        plt.show()


if __name__ == "__main__":
    NUM_EPOCHS = 250
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    LATENT_SPACE_SIZE = 16
    TRAIN_PATH = 'images/samples/train'
    TEST_PATH = 'images/samples/test'
    NETWORK_SAVE_PATH = 'models/pomdp_autoencoder_sz'+str(LATENT_SPACE_SIZE)+'.pth'

    #
    # Training + testing
    #
    p = transforms.Compose([transforms.Scale((100, 100))])
    trainloader = DataLoader(AE_Data(0, 800, dataset_path=TRAIN_PATH, transform=p), shuffle=True, batch_size=BATCH_SIZE)
    testloader = DataLoader(AE_Data(800, 200, dataset_path=TEST_PATH, transform=p), shuffle=True, batch_size=1)
    training(trainloader, testloader, NUM_EPOCHS, get_device(), NETWORK_SAVE_PATH)

    #
    # Playing around...
    #
    plot_encoding_comparisons(TEST_PATH, NETWORK_SAVE_PATH)


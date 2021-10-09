import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision.io import read_image


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=30000, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=256)
        self.enc3 = nn.Linear(in_features=256, out_features=128)
        self.enc4 = nn.Linear(in_features=128, out_features=64)
        self.enc5 = nn.Linear(in_features=64, out_features=32)
        self.enc6 = nn.Linear(in_features=32, out_features=16)
        self.enc7 = nn.Linear(in_features=16, out_features=4)
        # decoder
        self.dec0 = nn.Linear(in_features=4, out_features=16)
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
        x = F.relu(self.enc7(x))
        return x

    def decode(self, x):
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
            new_model.load_state_dict(torch.load("./models/pomdp_autoencoder_sz4.pth", map_location=torch.device('cpu')))
        if option == 'save':
            torch.save(new_model.state_dict(), "./models/pomdp_autoencoder_new.pth")
        # Change the device to GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = new_model.to(self.device)

        self._labels = []
        self.num_features = 4
        for i in np.arange(1, self.num_features+1):
            self._labels.append("f"+str(i))
        self._type = "ae"

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
        #sigmoid = nn.Sigmoid()
        #print(sigmoid(features))
        #norm_sig = sigmoid(features).numpy()
        #sum_r = norm_sig.sum(axis=1)
        #norm_sig = norm_sig / sum_r[:, np.newaxis]
        #print(norm_sig)
        return features/np.sum(features)


import cv2.cv2
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
        # encoder
        self.enc1 = nn.Linear(in_features=30000, out_features=16384)
        self.enc2 = nn.Linear(in_features=16384, out_features=8192)
        self.enc3 = nn.Linear(in_features=8192, out_features=4096)
        self.enc4 = nn.Linear(in_features=4096, out_features=2048)
        self.enc5 = nn.Linear(in_features=2048, out_features=1024)
        self.enc6 = nn.Linear(in_features=1024, out_features=512)
        self.enc7 = nn.Linear(in_features=512, out_features=256)
        self.enc8 = nn.Linear(in_features=256, out_features=128)
        self.enc9 = nn.Linear(in_features=128, out_features=64)
        self.enc10 = nn.Linear(in_features=64, out_features=32)
        self.enc11 = nn.Linear(in_features=32, out_features=16)

        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=512)
        self.dec6 = nn.Linear(in_features=512, out_features=1024)
        self.dec7 = nn.Linear(in_features=1024, out_features=2048)
        self.dec8 = nn.Linear(in_features=2048, out_features=4096)
        self.dec9 = nn.Linear(in_features=4096, out_features=8192)
        self.dec10 = nn.Linear(in_features=8192, out_features=16384)
        self.dec11 = nn.Linear(in_features=16384, out_features=30000)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        x = F.relu(self.enc7(x))
        x = F.relu(self.enc8(x))
        x = F.relu(self.enc9(x))
        x = F.relu(self.enc10(x))
        x = F.relu(self.enc11(x))
        return x

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = F.relu(self.dec7(x))
        x = F.relu(self.dec8(x))
        x = F.relu(self.dec9(x))
        x = F.relu(self.dec10(x))
        x = F.relu(self.dec11(x))
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
            img_path = dataset_path+'/img_'+str(img_start_idx+img_num)+'.jpg'
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


def train(net, trainloader, testloader, epochs, criterion, device, learning_rate):
    train_loss = []
    valid_loss = []

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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
        validation = validate(net, testloader, criterion, device)
        valid_loss.append(validation)
        print('Epoch {} of {}, Train Loss: {:.3f}, Valid Loss: {:.3f}'.format(
            epoch + 1, epochs, loss, validation))
        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch, decoded=1)
            save_decoded_image(img_orig.cpu().data, epoch, decoded=0)
    return train_loss, valid_loss


def training(trainloader, testloader, epochs, device, save_path, learning_rate):
    # create & load the neural network onto the device
    net = Autoencoder()
    net.to(device)
    make_dir()
    criterion = nn.MSELoss()
    # train the network
    train_loss, valid_loss = train(net, trainloader, testloader, epochs, criterion, device, learning_rate)
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
        img = cv2.imread(img_dir+'/img_'+str(idx)+'.jpg')
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

def start_train():
    NUM_EPOCHS = 250
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    LATENT_SPACE_SIZE = 16
    TRAIN_PATH = 'models/training_data'
    TEST_PATH = 'models/testing_data'
    NETWORK_SAVE_PATH = 'models/pomdp_ae_sz'+str(LATENT_SPACE_SIZE)+'.pth'

    #
    # Training + testing
    #
    p = transforms.Compose([transforms.Scale((100, 100))])
    trainloader = DataLoader(AE_Data(0, 100000, dataset_path=TRAIN_PATH, transform=p), shuffle=True, batch_size=BATCH_SIZE)
    testloader = DataLoader(AE_Data(0, 20000, dataset_path=TEST_PATH, transform=p), shuffle=True, batch_size=1)
    training(trainloader, testloader, NUM_EPOCHS, get_device(), NETWORK_SAVE_PATH, LEARNING_RATE)

    #
    # Playing around...
    #
    plot_encoding_comparisons(TEST_PATH, NETWORK_SAVE_PATH)

def gen_training_data():
    data_points = 20000
    data_dir = 'models/testing_data/'
    img_fname = '../images/airstrip_hand_segmented.png'
    im = cv2.imread(img_fname)
    h, w, _ = im.shape
    r = 75
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for i in range(data_points):
        if i % 1000 == 0:
            print("iteration ", i)
        x = np.random.randint(0+r, w-r)
        y = np.random.randint(0+r, h-r)
        sample = im[y-r:y+r,x-r:x+r]
        #plt.imshow(sample)
        #plt.show()
        cv2.imwrite(data_dir+"img_"+str(i)+".jpg", cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))



def decode_img():
    import cv2
    save_path = 'models/pomdp_autoencoder_sz16.pth'

    device = 'cpu'#get_device()
    net = Autoencoder()
    net.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    net.to(device)
    #other
    x = torch.tensor([0.10406552255153656, 0.5734096050262452, 0.32167674899101256, 1e-05, 0.014076241850852966, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 0.10211745351552963, 1e-05, 1e-05, 0.4699282765388489, 0.1446827232837677])
    #building
    x = torch.tensor([5.611148545410156, 40.754514786865236, 38.24892883300781, 1e-05, 1e-05, 1e-05, 1e-05, 37.147728729248044, 1e-05, 1.7446294280548096, 35.64239311218262, 12.73976674079895, 1e-05,1e-05, 39.2167423248291, 55.322322082519534])
    #road
    x = torch.tensor([1e-05, 1e-05, 5.555718803405762, 1.0793159246444701, 1e-05, 0.38576561536979675, 1e-05, 2.032072591781616, 1e-05, 1e-05, 17.133995628356935, 32.40111312866211, 1e-05, 1e-05, 6.844200801849365, 34.5301513671875])
    #corners
    x = torch.tensor([4.214598454101562, 23.07781505041504, 18.18980860710144, 0.6877232442932129, 1e-05, 1e-05, 1e-05, 16.424647045135497, 1e-05, 0.42366331921386724, 22.659137153625487, 12.249711794189455, 1e-05, 1e-05, 21.529578685760498, 30.49309959411621])
    # road edges
    x = torch.tensor([0.029287261827468873, 0.04013599859046936, 2.3334914749183655, 4.955508420135498, 1e-05, 2.342380830909729, 1e-05, 2.217990652519226, 1e-05, 0.5940385260162354, 13.246929794549942, 10.749766969680786, 1e-05, 0.09859951244163515, 2.3827168383178714, 12.619472789764405])

    # other
    x = torch.tensor([0.10405753783867747, 0.573409637070381, 0.3217064159872334, 1.2894505425979808e-5, 0.014070816017442018, 1.2837307175997797e-5, 1.288814715998603e-5, 1.293238963860023e-5, 1.2917557675344428e-5, 1.2890573516450871e-5, 1.282526220627611e-5, 0.10211438511031747, 1.2986478994099664e-5, 1.2809782434682077e-5, 0.46991749235737224, 0.14466905583150289])
    # bldg
    x = torch.tensor([6.089469291738582, 42.142100798351066, 38.47050837203591, 1.2830330378135523e-5, 1.28880568459174e-5, 1.2863349359832881e-5, 1.2861203820591458e-5, 37.38182348870205, 1.2863526479996833e-5, 1.8897802280833835, 35.71525333317329, 14.688479160820311, 1.2830902244937408e-5, 1.2870634475900679e-5, 39.18843514076485, 55.44318695958401])
    # road
    x = torch.tensor([1.2756886633819987e-5, 1.2869860733328047e-5, 5.556645489755599, 1.079760271721953, 1.2871504723331596e-5, 0.3986596480412641, 1.2968298096453702e-5, 2.030632230410035, 1.2788260593590503e-5, 1.2910060248468766e-5, 17.074240851070996, 32.39709797908506, 1.2894808764402497e-5, 1.2749692536372591e-5, 6.824755451837076, 34.529502494627295])
    # road edges
    x = torch.tensor([0.05937556948001951, 0.08096930838488206, 3.192177174459472, 5.085534070955735, 1.2827992816477074e-5, 2.480465495104383, 1.2848868292349633e-5, 4.007344825307063, 1.2851898989911436e-5, 0.8557131409154042, 17.02481140033259, 12.231961371416196, 1.2789281733785743e-5, 0.19871268985940438, 3.580727683944192, 15.304251258804795])
    # road intersections
    x = torch.tensor([1.3015602241589956e-5, 1.285566756368652e-5, 5.294120297966607, 4.579230212802138, 1.2909843444391086e-5, 2.539616874202805, 1.284262537826801e-5, 3.415115197555933, 1.2946060186206041e-5, 0.021964308516263308, 11.314151957355394, 20.50628059577887, 1.2887408764674975e-5, 1.2794984073077808e-5, 5.703622518990694, 22.33288200583781])

    x = torch.tensor([13.544604728406348, 43.39434225392947, 23.909831913163888, 0.4045392192066996, 1.2808298326978289e-5, 1.2905587962171901e-5, 1.2976839441753956e-5, 23.07947790715444, 1.285911895815119e-5, 0.9743737008036492, 33.6261699284891, 20.247337268707682, 1.286366725217337e-5, 1.3010158437401693e-5, 36.61769975788111, 38.75402166119687])
    outputs = net.decode(x)
    outputs = outputs.view(1, 3, 100, 100).cpu().data
    outputs = torch.squeeze(outputs)
    outputs = outputs.permute(1, 2, 0)
    outputs = cv2.cvtColor(outputs.numpy(), cv2.COLOR_BGR2RGB)
    plt.imshow(outputs)
    plt.show()


if __name__ == "__main__":
    #gen_training_data()
    start_train()
    #plot_encoding_comparisons('models/testing_data/', 'models/pomdp_ae_sz16.pth')


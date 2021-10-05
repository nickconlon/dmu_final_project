import torch
from torch import nn
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)

        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
        self.fcEnd = nn.Linear(4096, 3)
        self.norms = nn.Softmax(dim=1)

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.fcEnd(out)
        out = self.norms(out)
        return out


class Extractor:
    def __init__(self, option=None):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(512),
            transforms.Resize(448),
            transforms.ToTensor()
        ])
        model = models.vgg16(pretrained=True)
        new_model = FeatureExtractor(model)
        #
        if option == 'load':
            new_model.load_state_dict(torch.load("feature_extractor.pth"))
        if option == 'save':
            torch.save(new_model.state_dict(), "feature_extractor.pth")
        # Change the device to GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = new_model.to(self.device)

    def extract(self, image_data, r):
        transformed_data = self.transform(image_data)
        img = transformed_data.reshape(1, 3, 448, 448)
        img = img.to(self.device)

        with torch.no_grad():
            # Extract the feature from the image
            features = self.model(img)

        features = features.cpu().detach().numpy().reshape(-1)

        return features


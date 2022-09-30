# Importing libraries
import pandas as pd
import json
from PIL import Image
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Experiment setting
BATCH = 32  # Batch size
EPOCHS = 20  # total number of epochs

LR = 0.0001  # learning rate
IM_SIZE = 256  # image size

# Device setting
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths for train and test images
TRAIN_DIR = './Mydata/CASSAVA/train_images/'
TEST_DIR = './Mydata/CASSAVA/test_images/'

# Train and test data and corresponding Labels for train images
labels = json.load(open("./Mydata/CASSAVA/label_num_to_disease_map.json"))
train = pd.read_csv('./Mydata/CASSAVA/train.csv')
X_Train, Y_Train = train['image_id'].values, train['label'].values
X_Test = [name for name in (os.listdir(TEST_DIR))]

# Method for getting data
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.lbs = Labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        if "train" in self.dir:
            return self.transform(x), self.lbs[index]
        elif "test" in self.dir:
            return self.transform(x), self.fnames[index]

# Data loading
Transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IM_SIZE, IM_SIZE)),
     transforms.RandomRotation(90),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2)

testset = GetData(TEST_DIR, X_Test, None, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Model, criterion, and optimizer
model = torchvision.models.resnet152()
model.fc = nn.Linear(2048, 5, bias=True)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training stage
for epoch in range(EPOCHS):
    tr_loss = 0.0

    model = model.train()

    for i, (images, labels) in enumerate(trainloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.detach().item()

    model.eval()
    print('Epoch: %d | Loss: %.4f' % (epoch, tr_loss / i))

# Evaluation stage
s_ls = []
with torch.no_grad():
    model.eval()
    for image, fname in testloader:
        image = image.to(DEVICE)

        logits = model(image)
        ps = torch.exp(logits)
        _, top_class = ps.topk(1, dim=1)

        for pred in top_class:
            s_ls.append([fname[0], pred.item()])

# Creating submission data
submission = pd.DataFrame.from_records(s_ls, columns=['image_id', 'label'])
a = 1

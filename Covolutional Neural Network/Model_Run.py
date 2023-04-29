from superpixel.EdgeFlow import PixelBasedEdgeFlow
from superpixel.ImageProcessor import ImageProcessor
from torch.utils.data import DataLoader
from models import superpixel_GCN, superpixel_GAT
from torchvision import datasets
import torch
from superpixel.t1 import VaeTestDataset


EPOCHS=20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
superpixel_size = 35
dataset = datasets.MNIST
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RandomBasedEdgeFlow
imageprocessor = ImageProcessor
# imageprocessor = OrientatedImageProcessor
full_dataset = 1548
train_set = 1340
val_set = 208
test_set = 340
output_size = 10

model = superpixel_GAT
processor_type, NN = model

train_dataset = VaeTestDataset('C:/Users/Yang Zhao/Desktop/2023/Graph review paper/OGM-datasets/OGM-Turtlebot2/SGAN_train', "train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

test_dataset = VaeTestDataset('C:/Users/Yang Zhao/Desktop/2023/Graph review paper/OGM-datasets/OGM-Turtlebot2/SGAN_test', "test")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

#val_dataset = VaeTestDataset('C:/Users/Yang Zhao/Desktop/2023/Graph review paper/OGM-datasets/OGM-Turtlebot2/SGAN_val', "val")
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=True)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    num_batches = int(len(train_dataset) / train_loader.batch_size)
    for batch_idx, batch in tqdm(enumerate(train_loader), total=num_batches):
        scans = batch['scan']
        scans = scans.to(device)
        optimizer.zero_grad()
        output = model(scans)
        #loss = F.nll_loss(output, target)
        #loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(scans), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for scans in test_loader:
            scans = scans.to(device)
            output = model(scans)
            test_loss += F.nll_loss(output, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_dataloader, optimizer, epoch)
    test(model, DEVICE, test_dataloader)
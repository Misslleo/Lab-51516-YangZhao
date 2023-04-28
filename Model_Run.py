from superpixel.EdgeFlow import PixelBasedEdgeFlow
from superpixel.ImageProcessor import ImageProcessor
from torch.utils.data import DataLoader
from constants import DEVICE
from models import superpixel_GCN, superpixel_GAT
from torchvision import datasets
import torch
import numpy as np
from datetime import timedelta
import time
from superpixel.SuperpixelDataset import SuperpixelSCDataset
from superpixel.SuperpixelDataset1 import SuperpixelSCDataset1
from superpixel.SuperpixelDataset2 import SuperpixelSCDataset2

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

train_data = SuperpixelSCDataset('C:/Users/Yang Zhao/PycharmProjects/rob/outputt/train', 'train', superpixel_size,
                                 edgeFlow, processor_type, imageprocessor, train_set)
test_data = SuperpixelSCDataset1('C:/Users/Yang Zhao/PycharmProjects/rob/outputt/test', 'test', superpixel_size,
                                 edgeFlow,processor_type, imageprocessor, test_set)
val_data = SuperpixelSCDataset2('C:/Users/Yang Zhao/PycharmProjects/rob/outputt/val', 'val', superpixel_size, edgeFlow,
                                processor_type, imageprocessor, val_set)
train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                           shuffle=True, pin_memory=True)
test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                          shuffle=True, pin_memory=True)
val_dataset = DataLoader(val_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                         shuffle=True, pin_memory=True)


def run(processor_type, NN):
    optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    average_time, train_acc, val_acc = train(NN, 100, train_dataset, val_dataset, optimizer, criterion, processor_type)
    output = average_time.txt


def train(NN, epoch_size, train_data_, val_data_, optimizer, criterion, processor_type):
    train_running_loss = 0
    t = 0
    best_val_acc = 0
    for epoch in range(epoch_size):
        t1 = time.perf_counter()
        epoch_train_running_loss = 0
        train_acc, training_acc = 0, 0
        val_acc, validation_acc = 0, 0
        i, j = 0, 0
        NN.train()
        for simplicialComplex, train_labels in train_data_:
            simplicialComplex.to_device()
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            optimizer.zero_grad()
            prediction = NN(simplicialComplex)
            loss = criterion(prediction)
            loss.backward()
            optimizer.step()
            epoch_train_running_loss += loss.detach().item()
            train_acc = (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
            training_acc += (train_acc - training_acc) / i
        t2 = time.perf_counter()
        NN.eval()
        for simplicialComplex, val_labels in val_data_:
            simplicialComplex.to_device()
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            prediction = NN(simplicialComplex)
            val_acc = (torch.argmax(prediction, 1).flatten() == val_labels).type(torch.float).mean().item()
            j += 1
            validation_acc += (val_acc - validation_acc) / j
        t = (t * epoch + (t2 - t1)) / (epoch + 1)
        epoch_train_running_loss /= i
        train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
        if validation_acc > best_val_acc:
            torch.save(NN.state_dict(), f'./data/{NN.__class__.__name__}_nn.pkl')
            best_val_acc = validation_acc
            associated_training_acc = training_acc
        print(
            f"Epoch {epoch}"
            f"| Train accuracy {training_acc:.4f} | Validation accuracy {validation_acc:.4f}")
    return t, associated_training_acc, best_val_acc


if __name__ == "__main__":
    processor_type, NN = model
    if NN in {superpixel_GCN[1], superpixel_GAT[1]}:
        NN = NN(5, output_size).to(DEVICE)
    else:
        NN = NN(5, 10, 15, output_size).to(DEVICE)

    run(processor_type, NN.to(DEVICE))
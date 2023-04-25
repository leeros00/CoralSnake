import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from typing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset1D(Dataset):
    def __init__(self, signals, segmentation_maps) -> None:
        super().__init__()
        self.signals = signals
        self.segmentation_maps = segmentation_maps
    

    def __len__(self):
        return len(self.signals)
    

    def __getitem__(self, index):
        signal = self.signals[index]
        segmentation_map = self.segmentation_maps[index]
        return signal, segmentation_map
    


class CoralSnakeUNet(nn.Module):
    def __init__(self, in_channels: int=1, out_channels: int=1, init_features: int=32) -> None:
        super().__init__()

        features = init_features

        self.encoder1 = nn.Sequential(nn.Conv1d(in_channels, features, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features, features, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features),
                                      nn.ReLU(inplace=True),)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(nn.Conv1d(features, features*2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features*2, features*2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*2),
                                      nn.ReLU(inplace=True),)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = nn.Sequential(nn.Conv1d(features*2, features*4, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*4),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features*4, features*4, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*4),
                                      nn.ReLU(inplace=True),)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = nn.Sequential(nn.Conv1d(features*4, features*8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*8),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features*8, features*8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*8),
                                      nn.ReLU(inplace=True),)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(nn.Conv1d(features*8, features*16, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm1d(features*16),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(features*16, features*16, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm1d(features*16),
                                        nn.ReLU(inplace=True),
                                        nn.ConvTranspose1d(features*16, features*8, kernel_size=2, stride=2),)
        self.decoder4 = nn.Sequential(nn.Conv1d(features*16, features*8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*8),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features*8, features*8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*8),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose1d(features*8, features*4, kernel_size=2, stride=2),)
        self.decoder3 = nn.Sequential(nn.Conv1d(features*8, features*4, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*4),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features*4, features*4, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*4),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose1d(features*4, features*2, kernel_size=2, stride=2),)
        self.decoder2 = nn.Sequential(nn.Conv1d(features*4, features*2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features*2, features*2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features*2),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose1d(features*2, features, kernel_size=2, stride=2),)
        self.decoder1 = nn.Sequential(nn.Conv1d(features*2, features, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(features, features, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm1d(features),
                                      nn.ReLU(inplace=True),)
        self.out_conv = nn.Conv1d(features, out_channels, kernel_size=1, stride=1)
    

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool1(enc2))
        enc4 = self.encoder4(self.pool1(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.decoder4(torch.cat((bottleneck, enc4), dim=1))
        dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
        dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
        dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))
        out = self.out_conv(dec1)
        return out
    

def generate_signal_data(num_samples=1000, error_perc=0.07,
                         x_min=650, x_max=850, signal_length=992):
    signals=[]
    annotation_maps = []
    plottable_segmentation_maps = []
    vertical_stretches = np.linspace(1, 100, num_samples)
    error = np.random.rand(num_samples)*error_perc

    for i in range(num_samples):
        x = np.linspace(-np.pi, np.pi, signal_length+i)
        signal = vertical_stretches[i]*np.sin(x)
        if i % 2 == 0:
            error_factor= 1 + error
        else:
            error_factor= 1 - error
        x_min_error = round(error_factor[i]*x_min)
        x_max_error = round(error_factor[i]*x_max)
        signals.append(signal)

        annotation_map = np.zeros(signal_length+i)
        annotation_map[375:600] = 1
        annotation_map[x_min_error:x_max_error] = 1
        annotation_maps.append(annotation_map)

        plottable_segmentation_map = np.full(signal_length+i, np.nan)
        plottable_segmentation_map[x_min_error:x_max_error] = signal[x_min_error:x_max_error]
        plottable_segmentation_maps.append(plottable_segmentation_map)

    return signals, annotation_maps, plottable_segmentation_maps


signals, annotation_maps, plottable_segmentation_maps = generate_signal_data()


def pad_data(signals, annotation_maps):
    max_len_signals = max([len(signal) for signal in signals])
    max_len_annotation_maps = max([len(annotation_map) for annotation_map in annotation_maps])

    padded_signals = np.zeros((len(signals), max_len_signals))
    padded_annotation_maps = np.zeros((len(annotation_maps), max_len_annotation_maps))
    # assert that they are the same
    for i, signal in enumerate(signals):
        padded_signals[i, :len(signal)] = signal

    for j, annotation_map in enumerate(annotation_maps):
        padded_annotation_maps[j, :len(annotation_map)] = annotation_map
    
    #original_lengths = np.array([len(signal) for signal in signals])
    
    return padded_signals, padded_annotation_maps

padded_signals, padded_annotation_maps = pad_data(signals=signals, annotation_maps=annotation_maps)


def convert_arrays_to_tensors(signals, annotation_maps):
    tensor_signals = []
    tensor_annotation_maps = []

    for i in range(len(signals)):
        signal = signals[i]
        tensor_signal = torch.from_numpy(signal)
        tensor_signal.resize_(1, 1984) # change it to use some desired length specified in the args
        tensor_signals.append(tensor_signal)
        
        annotation_map = annotation_maps[i]
        tensor_annotation_map = torch.from_numpy(annotation_map)
        tensor_annotation_map.resize_(1, 1984)
        tensor_annotation_maps.append(tensor_annotation_map)
    return tensor_signals, tensor_annotation_maps


def train(signal_train, label_train, batch_size=32, lr=0.001, num_epochs=20, model_name='1d_unet.pt', device=device):
    train_dataset = Dataset1D(signals=signal_train, segmentation_maps=label_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    model = CoralSnakeUNet(out_channels=1).to(device, dtype=torch.float64)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for signals, segmentation_maps in train_dataloader:
            outputs = model(signals)
            loss = criterion(outputs, segmentation_maps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1} loss: {loss.item()}')
    torch.save(model.state_dict(), model_name)



signal_train, signal_test, label_train, label_test = train_test_split(padded_signals, padded_annotation_maps, test_size=0.20, random_state=22)
signal_train, label_train = convert_arrays_to_tensors(signals=signal_train, annotation_maps=label_train)
signal_test, label_test = convert_arrays_to_tensors(signals=signal_test, annotation_maps=label_test)


train(signal_train=signal_train, label_train=label_train)








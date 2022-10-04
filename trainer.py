import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from os import path
from pathlib import Path

from dataset import MNIST

class Trainer:
    def __init__(self, args):
        # init
        self.args = args
        self.model_name = args.model
        self.n_colors = args.n_colors
        self.n_classes = args.n_classes
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device_ids = args.device_ids
        
        # model select and modification
        if(self.model_name == 'resnet50'):
            self.model = timm.create_model('resnet50', pretrained=True)
            self.model.conv1 = nn.Conv2d(self.n_colors, self.model.conv1.out_channels, 
                                        kernel_size=self.model.conv1.kernel_size, stride=self.model.conv1.stride, 
                                        padding=self.model.conv1.padding, bias=self.model.conv1.bias)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)
        else:
            print('no implement of {}'.format(self.model_name))

        self.model = nn.DataParallel(self.model,device_ids=self.device_ids) # 指定要用到的设备
        self.model = self.model.cuda(device=self.device_ids[0]) # 模型加载到设备0，这里只是定义一个样式

        # dataset
        self.dataset = MNIST()
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # loss
        self.train_loss = []
        self.train_acc = []
        self.train_process_path = path.join('trainProcess', self.model_name)
        self.train_loss_file = self.model_name+'_train_loss.npy'
        self.train_acc_file = self.model_name+'train_acc.npy'

        # save path
        self.save_path = path.join('trainedModel', self.model_name)
        self.file_name = self.model_name+'_parameter.pkl'

    def train(self):
        for epoch in range(self.epochs):
            print('Epoch {}/{} start'.format(epoch+1, self.epochs))
            running_loss = 0.0
            for x_train, y_train in tqdm(self.dataLoader):
                x_train, y_train = x_train.cuda(device=self.device_ids[0]), y_train.cuda(device=self.device_ids[0])
                outputs = self.model(x_train)
                
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.train_loss.append(running_loss)
            print('Epoch {}/{}, train loss:{}'.format(epoch+1, self.epochs, running_loss))
        
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path.join(self.save_path, self.file_name))
        Path(self.train_process_path).mkdir(parents=True, exist_ok=True)
        np.save(path.join(self.train_process_path, self.train_loss_file), self.train_loss)

if __name__ == '__main__':
    from option import args
    trainer = Trainer(args)
    trainer.train()


        
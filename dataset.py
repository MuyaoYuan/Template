import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor, ToPILImage

class MNIST(Dataset):
    def __init__(self, trainFlag=True, transform=ToTensor()):
        super().__init__()
        self.trainFlag = trainFlag
        self.transform = transform
        if self.trainFlag:
            self.train = pd.read_csv('/data/ymy/MNIST/train.csv')
            self.y = np.asarray(self.train['label'])
            self.X = np.asarray(self.train.drop('label', axis=1))
        else:
            self.test = pd.read_csv('/data/ymy/MNIST/test.csv')
            self.X = np.asarray(self.test)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if self.trainFlag:
            x_item = self.X[index].reshape(28,28) / 255
            y_item = self.y[index]
            return self.transform(x_item).float(), torch.tensor(y_item)
        else:
            x_item = self.X[index].reshape(28,28)
            return self.transform(x_item).float()

if __name__ == '__main__':
    # mnist = MNIST()
    
    # dataLoader = DataLoader(mnist, batch_size=10, shuffle=True)
    # dataIter = iter(dataLoader)
    # input, label = dataIter.next()
    
    # transform = ToPILImage()
    # img = transform(input[0])
    # img.save('test/datasetTest.png')
    # print(label[0])

    mnist = MNIST(trainFlag=False)
    
    dataLoader = DataLoader(mnist, batch_size=20, shuffle=False)
    dataIter = iter(dataLoader)
    input = dataIter.next()
    
    transform = ToPILImage()
    img = transform(input[2])
    img.save('test/datasetTest.png')

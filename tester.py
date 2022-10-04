import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from os import path
from pathlib import Path

from dataset import MNIST

class Tester:
    def __init__(self, args):
        # init
        self.args = args
        self.model_name = args.model
        self.n_colors = args.n_colors
        self.n_classes = args.n_classes
        self.batch_size = args.batch_size
        self.device_ids = args.device_ids
        
        # model select
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

        # save path of model
        self.save_path = path.join('trianedModel', self.model_name)
        self.file_name = self.model_name+'_parameter.pkl'

        # model reload
        self.model.load_state_dict(torch.load(path.join(self.save_path, self.file_name)))

        # dataset
        self.dataset = MNIST(trainFlag=False)
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # result
        self.result = np.array([])

        # save path of result
        self.result_path = path.join('result', self.model_name)
        self.result_name = self.model_name+'_submission.csv'

    def test(self):
        for x_test in tqdm(self.dataLoader):
            x_test = x_test.cuda(device=self.device_ids[0])
            outputs = self.model(x_test)
            outputs = torch.argmax(outputs, dim=1)
            outputs = outputs.cpu().detach().numpy()
            self.result = np.append(self.result, outputs)
            
        imageId = np.arange(1, len(self.dataset)+1)
        submission = np.stack([imageId, self.result], axis=1).astype(int)
        submission = pd.DataFrame(submission, columns=['ImageId', 'Label'])

        Path(self.result_path).mkdir(parents=True, exist_ok=True)
        submission.to_csv(path.join(self.result_path, self.result_name),index=False)

if __name__ == '__main__':
    from option import args
    Tester = Tester(args)
    Tester.test()


        
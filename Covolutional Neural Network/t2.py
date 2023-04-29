# 数据处理
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

#定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self,root):
    # 所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

'''''
if __name__ == '__main__':
        dataSet=FlameSet('C:/Users/Yang Zhao/PycharmProjects/rob/outputt/train')
        print(dataSet[0])
'''''




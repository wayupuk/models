import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomUnetDataset(Dataset):
    def __init__(self, annotations_file,train,img_dir=None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.method = train
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if self.method == True:
            self.img_labels = self.img_labels.iloc[:int(len(self.img_labels)*0.7)].reset_index(drop=True)
        elif self.method == False:
            self.img_labels = self.img_labels.iloc[int(len(self.img_labels)*0.7):].reset_index(drop=True)
        else:
            raise "condition incorrect"

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # label_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])        
        img_path = self.img_labels.iloc[idx, 0]
        label_path = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)
        label = read_image(label_path)
        self.resize_transform = transforms.Resize((224, 224))
        image = self.resize_transform(image)
        label = self.resize_transform(label)
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label
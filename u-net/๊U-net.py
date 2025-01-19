### U-net implement
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import cv2
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets
from dataset import CustomUnetDataset
from torch.utils.data import DataLoader
import logging
Epoch = 20
batch = 4
learning_rate = 1e-5
save_step = 1
weight_decay = 1e-8
momentum = 0.99
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.info(f'''
             start training
             Epoch {Epoch}
             Batch size {batch}
             Lr {learning_rate}
             save_step {save_step}
             weight_decay {weight_decay}
             momentum {momentum}
             ''')
# print(cv2.imread("ipyfiles/food-ex2.jpg").shape)
# Example image size = (800,641,3) w,h,c

# torch.concat dim 0 tensor will connect to tthe last low but 1 will plus ion every row
# stupidest way to make u-net
class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)
        return x1
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.downsample1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
            
        )
        self.downsample2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
            
        )
        self.downsample3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,256,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
            
        )
        self.downsample4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512,512,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
            
        )
        
        ### to make it equivalent
        self.bottom = nn.Sequential(
                        nn.Conv2d(512,1024,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(1024,1024,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
            
        )
        self.upsampling4 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512,512,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
        )
        
        self.upsampling3 = nn.Sequential(
                        nn.Conv2d(512,256,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,256,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
        )
        self.upsampling2 = nn.Sequential(
                        nn.Conv2d(256,128,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
        )
        self.upsampling1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,kernel_size=3,padding=1)
                        ,nn.ReLU(inplace=True),
        )
        self.last = nn.Sequential(nn.Conv2d(64,3,kernel_size=1))
        self.up1  = UpSample(1024)
        self.up2  = UpSample(512)
        self.up3  = UpSample(256)
        self.up4  = UpSample(128)
    def forward(self,x):
        
        x1 = self.downsample1(x)
        x1m = self.maxpool(x1)
        
        x2 = self.downsample2(x1m)
        x2m = self.maxpool(x2)
        
        x3 = self.downsample3(x2m)
        x3m = self.maxpool(x3)
        
        x4 = self.downsample4(x3m)
        x4m = self.maxpool(x4)
        
        b = self.bottom(x4m)
        
        y1 = self.up1(b)
        y1 = torch.cat([x4,y1],dim=1)
        y2 = self.upsampling4(y1)
        
        y2 = self.up2(y2)
        y2 = torch.cat([x3,y2],dim=1)
        y3 = self.upsampling3(y2)
        
        
        y3 =  self.up3(y3)
        y3 = torch.cat([x2,y3],dim=1)
        y4 = self.upsampling2(y3)
        
        y4 =  self.up4(y4)
        
        y4 = torch.cat([x1,y4],dim=1)
        y5 = self.upsampling1(y4)
        
        y = self.last(y5)
        return y


train = CustomUnetDataset("ipyfiles/u-net/Unet_train.csv",img_dir="ipyfiles/u-net/",train=True)
eval = CustomUnetDataset("ipyfiles/u-net/Unet_train.csv",img_dir="ipyfiles/u-net/",train=False)
train_set = DataLoader(train,batch_size=batch,shuffle=True)
eval_set = DataLoader(eval,batch_size=batch,shuffle=True)


model = Unet().to("cuda")
criterion = nn.BCEWithLogitsLoss()
optimizer  = torch.optim.RMSprop(model.parameters(),lr = learning_rate, weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
for name, param in model.named_parameters():
    if param.grad is None:
        param.requires_grad = True


for epoch in (range(1,Epoch+1)):
    loss = 0
    total_loss = 0
    eval_loss = 0
    print("Epoch number: ","-----",epoch ,"-----")
    for batch in tqdm(train_set,total=len(train_set)):
        optimizer.zero_grad()
        model.train()
        img = (batch[0]/255).to("cuda")
        label = (batch[1]/255).to("cuda")
        output = model(img)
        loss = criterion(output,label)
        total_loss+= loss.item()
        
        loss.backward()
        optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
    print("trained_loss: ",total_loss/len(train_set),f"Lr: {current_lr}")
    for batch in tqdm(eval_set,total=len(eval_set)):
        model.eval()
        img = (batch[0]/255).to("cuda")
        label = (batch[1]/255).to("cuda")
        output = model(img)
        val_loss = criterion(output,label)
        eval_loss += val_loss.item()
    print("eval_loss: ",eval_loss/len(eval_set),"\n\n")
    scheduler.step(loss)
    
    if epoch % save_step ==0:
        torch.save(model.state_dict(),f"./ipyfiles/u-net/trained/checkpoint-{epoch}")
        print("Chekcpoint {epoch} have been saved")
print("-----","Finished","-----")
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
# from keras.src.metrics import accuracy
# from openpyxl.styles.builtins import output
from tensorflow.python.keras.models import save_model
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import AverageMeter,accuracy
from model import model_dict
import numpy as np
import time
import random
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import shutil
from torch.utils.data import Dataset
#argparse############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model_names",type=str,default="resnet50")
parser.add_argument("--re_size",type=int,default="224")
parser.add_argument("--pre_trained",type=bool,default=False)
parser.add_argument("--classes_num",type=int,default=1)
parser.add_argument("--dataset",type=str,default="/neural_network/dataset_Reg")
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--epoch",type=int,default=300)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--weight-decay",type=float,default=1e-4)
parser.add_argument("--seed",type=int,default=33)
parser.add_argument("--gpu-id",type=int,default=0)
parser.add_argument("--print_freq",type=int,default=1)
parser.add_argument("--exp_postfix",type=str,default="result1")
parser.add_argument("--txt_name",type=str,default="lr0.01_wdSe-4")
args = parser.parse_args()
image_dir = "/mnt/d/prosessed_imag"
output_dir = "dataset_reg"
train_ratio = 0.8
file_path = '/home/liushuai/seed/seedprosessing/sorted_output.xlsx'
#####################################################################################


all_sheets = pd.read_excel(file_path,sheet_name=None,header=None)

ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],\
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']\
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = np.concatenate((ori_data[:, 1:7], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)
labels = data[:,0]
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))

if len(image_files) != len(labels):
    raise ValueError(f"image_files_len:{len(image_files)}labels_len{len(labels)}")

data_img = list(zip(image_files,labels))
train_data,test_data = train_test_split(data_img,train_size=train_ratio,random_state=42)
def clear_output_dirs(subsets=["train","test"]):
    for subset in subsets:
        dir_path = os.path.join(output_dir,subset)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def copy_files(data, subset):
    for filename,label in data:
        src = os.path.join(image_dir, filename)
        dst_dir = os.path.join(output_dir, subset)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        shutil.copy2(src, dst)

clear_output_dirs()
copy_files(train_data, "train")
copy_files(test_data, "test")


###########seed################
def seed_torch(seed=74):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
exp_name = args.exp_postfix
exp_path = "./model/{}/{}".format(args.model_names,exp_name)
os.makedirs(exp_path,exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transformer_train = transforms.Compose([
    #transforms.RandomRotation(90),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize((args.re_size,args.re_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.3738,0.3738,0.3738),(0.3240,0.3240,0.3240))
]
)
transformer_test = transforms.Compose([
    #transforms.RandomRotation(90),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize((args.re_size,args.re_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.3738,0.3738,0.3738),(0.3240,0.3240,0.3240))
]
)
class WeightDataset(Dataset):
    def __init__(self,image_dir,data_list,transform=None):
        self.image_dir = image_dir
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_name,weight = self.data_list[item]
        img_path = os.path.join(self.image_dir,img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        weight = torch.tensor(weight,dtype=torch.float32)
        return img,weight
train_dir = '/home/liushuai/seed/neural_network/dataset_reg/train'
test_dir = '/home/liushuai/seed/neural_network/dataset_reg/test'
trainset = WeightDataset(train_dir, train_data, transform=transformer_train)
testset = WeightDataset(test_dir, test_data, transform=transformer_train)

train_loader = DataLoader(trainset,batch_size=args.batch_size,num_workers=0,shuffle=True)
test_loader = DataLoader(testset,batch_size=args.batch_size,num_workers=0,shuffle=True)



def train_one_epoch(model,optimizer,train_loader):
    model.train()
    criterion = nn.MSELoss()
    loss_recorder = AverageMeter()

    for (inputs,targets) in tqdm(train_loader,desc='train'):
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        out = model(inputs)
        loss = criterion(out.squeeze(1), targets)

        loss_recorder.update(loss.item(),n=inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = loss_recorder.avg
    return losses

def evaluate(model,test_loader):
    model.eval()
    criterion = nn.MSELoss()
    loss_recorder = AverageMeter()
    with torch.no_grad():
        for inputs, label in tqdm(test_loader,desc="Evaluating"):
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            out = model(inputs)
            print('reslut:',out.squeeze(1))
            print('real:',label)
            loss = criterion(out.squeeze(1), label)
            loss_recorder.update(loss.item(), n=inputs.size(0))

        losses = loss_recorder.avg
        return losses

def train(model,optimizer,train_loader,test_loader,scheduler,tb_writer):
    since = time.time()
    min_losses = 100
    f = open(os.path.join(exp_path,"{}.txt".format(args.txt_name)),"w")

    for epoch in range(args.epoch):
        train_losses = train_one_epoch(
              model,optimizer,train_loader
        )
        test_losses = evaluate(model,test_loader)
        if min_losses > test_losses:
            min_losses = test_losses
            stat_dict = dict(epoch=epoch+1,model=model.state_dict(),acc=test_losses)
            name = os.path.join(exp_path,"regression","best.pth")
            os.makedirs(os.path.dirname(name),exist_ok=True)
            torch.save(stat_dict,name)
        scheduler.step()

        tags = ['train_losses',
                'test_losses'
                ]
        tb_writer.add_scalar(tags[0],train_losses,epoch+1)
        tb_writer.add_scalar(tags[1],test_losses,epoch+1)
        if (epoch+1) % args.print_freq == 0:
            msg = "epoch:{} model{} train loss:{:.2f} test_losses:{:.2f}\n".format(
                epoch+1,
                args.model_names,
                train_losses,
                test_losses
            )
            print(msg)
            f.write(msg)
            f.flush()
    msg_best = "model:{} min_loss:{:.2f}\n".format(args.model_names,min_losses)
    time_elapsed = "traning time:{}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


if __name__ == "__main__":
    tb_path = "model/{}/{}".format(args.model_names,args.exp_postfix)
    tb_writer = SummaryWriter(log_dir=tb_path)
    save_path = r"/neural_network/model/best_model.pth"
    model = model_dict[args.model_names](num_classes=args.classes_num,pretrained=args.pre_trained)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer,T_max=args.epoch)

    train(model,optimizer,train_loader,test_loader,scheduler,tb_writer)
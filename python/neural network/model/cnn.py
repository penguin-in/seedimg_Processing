import torch.nn as nn
out_channel = 16
kernel_size = 5


class simplecnn(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,out_channel,kernel_size=kernel_size,stride=1,padding=1),#input:3*512*512 output:16*512*512
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4),#input:16*512*512 output:16*128*128
            nn.Conv2d(out_channel,out_channel*2,kernel_size=kernel_size,stride=1,padding=1),#input:16*128*128 output:32*128*128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#input:32*128*128 output:32*64*64
            nn.Conv2d(out_channel*2, out_channel*(2**2), kernel_size=kernel_size, stride=1, padding=1),  # input:32*64*64 output:64*64*64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),# input:64*64*64 output:64*32*32
            nn.Conv2d(out_channel*(2**2), out_channel*(2**3), kernel_size=kernel_size, stride=1, padding=1),  # input:64*32*32 output:128*32*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input:128*32*32 output:128*16*16
            nn.Conv2d(out_channel*(2**3), out_channel*(2**4), kernel_size=kernel_size, stride=1, padding=1),  # input:128*16*16 output:256*16*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input:256*16*16 output:256*8*8
            nn.Conv2d(out_channel*(2**4), out_channel*(2**5), kernel_size=kernel_size, stride=1, padding=1),  # input:256*8*8 output:512*8*8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input:512*8*8 output:512*4*4
            nn.Conv2d(out_channel*(2**5), out_channel*(2**6), kernel_size=3, stride=1, padding=1),  # input:512*4*4 output:1024*4*4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input:1024*4*4 output:1024*2*2
            nn.Conv2d(out_channel * (2 ** 6), out_channel * (2 ** 6), kernel_size=2, stride=1, padding=1),# input:1024*2*2 output:1024*4*4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # input:1024*2*2 output:1024*1*1

        )

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,num_class)
        )
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

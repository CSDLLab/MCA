import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class Moment_efficient(nn.Module):  # 
    def forward(self, x):
        avg_x = torch.mean(x, (2,3)).unsqueeze(-1).permute(0,2,1)
        std_x = torch.std(x, (2,3), unbiased=False).unsqueeze(-1).permute(0,2,1)
        moment_x = torch.cat((avg_x, std_x), dim=1) # bs,*,c
        return moment_x
    
class Moment_Strong(nn.Module):
    def forward(self, x):
        
        # mean std
        n = x.shape[2] * x.shape[3]
        avg_x = torch.mean(x, (2,3), keepdim=True) # bs,c,1,1
        std_x = torch.std(x, (2,3), unbiased=False, keepdim=True) # bs,c,1,1
        # skew
        skew_x1 = 1/n * (x - avg_x)**3
        skew_x2 = std_x**3
        skew_x = torch.sum(skew_x1,(2,3), keepdim=True)/(skew_x2 + 1e-5) # bs,c,1,1
       
        avg_x = avg_x.squeeze(-1).permute(0,2,1)
        skew_x = skew_x.squeeze(-1).permute(0,2,1)

        
        moment_x = torch.cat((avg_x, skew_x), dim=1) # bs,*,c
        return moment_x 

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        k = 3  # for  Moment_Strong , k= 7  
        self.conv = nn.Conv1d(2, 1, kernel_size=k, stride=1, padding=(k-1)//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.conv(x)
        output = self.sigmoid(y)
        return output

class MomentAttention_v1(nn.Module):
    def __init__(self, **kwargs):
        super(MomentAttention_v1, self).__init__()
        self.moment = Moment_efficient()
        self.c = ChannelAttention()

    def forward(self, x):
        y = self.moment(x) #bs,2,c
        result = self.c(y) #bs,1,c
        result = result.permute(0,2,1).unsqueeze(-1) # bs,c,1,1
        return x*result.expand_as(x)



class MomentAttention_v2(nn.Module):
    def __init__(self, **kwargs):
        super(MomentAttention_v2, self).__init__()
        self.moment = Moment_Strong()
        self.c = ChannelAttention()

    def forward(self, x):
        y = self.moment(x) #bs,2,c
        result = self.c(y) #bs,1,c
        result = result.permute(0,2,1).unsqueeze(-1) # bs,c,1,1
        return x*result.expand_as(x)


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    mca = MomentAttention_v1()
    output=mca(input)
    print(output.shape)

    mca2 = MomentAttention_v2()
    output=mca2(input)
    print(output.shape)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DownBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
      super().__init__()
      self.convs = nn.Sequential( 
                                  nn.Conv2d(in_ch, out_ch, 3 , bias = False, padding = 1),
                                  nn.BatchNorm2d(out_ch, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(out_ch, out_ch, 3, bias = False, padding = 1),
                                  nn.BatchNorm2d(out_ch, affine=False),
                                  nn.ReLU(), 
                                  nn.MaxPool2d(2)                                          
                                  ).to(device)
  def forward(self, x):
    return self.convs(x)

class UpBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
      super().__init__()
      self.conv = nn.Sequential(
                                  nn.Conv2d(in_ch, out_ch, 3, padding = 0, bias = False),
                                  nn.BatchNorm2d(out_ch, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(out_ch, out_ch, 3, padding = 0, bias = False),
                                  nn.BatchNorm2d(out_ch, affine=False),
                                  nn.ReLU()
                                ).to(device)


  def forward(self, x):
    return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels[0], affine=False)
        self.blocks = nn.ModuleList([DownBlock(channels[i], channels[i+1]) 
                                        for i in range(len(channels) -1)])

    def forward(self, x):
      if self.channels[0] == 1:
        x = torch.mean(x, axis = 1, keepdim = True)

      x = self.norm(x)
      features = [x]

      for b in self.blocks:
        x = b(x)
        features.append(x)
      return features


class Decoder(nn.Module):
    def __init__(self, enc_ch, dec_ch):
        super().__init__()
        enc_ch = enc_ch[::-1]
        #print('c:', enc_ch)
        #print('d:', dec_ch)
        self.convs = nn.ModuleList( [UpBlock(enc_ch[i+1] + dec_ch[i], dec_ch[i+1]) 
                                                      for i in range(len(dec_ch) -2)] )
        self.conv_heatmap = nn.Sequential( 
                                        nn.Conv2d(dec_ch[-2], dec_ch[-2], 3, padding = 1, bias = False),
                                        nn.BatchNorm2d(dec_ch[-2], affine=False),
                                        nn.ReLU(),
                                        nn.Conv2d(dec_ch[-2], 1, 1),
                                        nn.Sigmoid()
                                     ).to(device)
        
    def forward(self, x):
        x = x[::-1]
        x_next = x[0]
        for i in range(len(self.convs)):
          upsampled = F.interpolate(x_next, size = x[i+1].size()[-2:], mode = 'bilinear', align_corners = True)
          x_next = torch.cat([upsampled, x[i+1]], dim = 1)
          #print(x_next.shape, '-->')
          x_next = self.convs[i](x_next)
          #print(x_next.shape)
        
        x_next = F.interpolate(x_next, size = x[-1].size()[-2:], mode = 'bicubic', align_corners = True)
        #print(x_next.shape)
        #print('-----------')
        return self.conv_heatmap(x_next)
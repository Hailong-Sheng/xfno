import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce

from .layer import MLPConv2d, SpectralConv2d

class FNO2d(nn.Module):
    """ 2D Fourier neural operator """
    def __init__(self, input_channel: int=1, output_channel: int=1, width: int=32, 
                 mode1_num: int=16, mode2_num: int=16, padding: int=8, layer_num: int=4, 
                 activation=nn.GELU(), input_normalizer=None, output_normalizer=None
                 ) -> None:
        """ initialization
        args:
            input_channel: number of input channels
            output_channel: number of output channels
            width: number of hidden channels
            mode1_num: number of Fourier modes in the 1th dimension
            mode2_num: number of Fourier modes in the 2th dimension
            padding: padding size
            layer_num: number of Fourier layers
            activation: activation function
            input_scale: scale of input
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.width = width
        self.mode1_num = mode1_num
        self.mode2_num = mode2_num
        self.padding = padding
        self.layer_num = layer_num
        self.activation = activation
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        
        # encoder layer
        self.encoder = nn.Linear(self.input_channel, self.width)
        
        # convolutional layer
        self.conv_layer = nn.ModuleList()
        self.spconv_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            self.conv_layer.append(nn.Conv2d(self.width, self.width, 1))
            self.spconv_layer.append(
                SpectralConv2d(self.width, self.width, self.mode1_num, self.mode2_num)
            )

        # decoder layer
        self.decoder = MLPConv2d(self.width, self.output_channel, self.width*4, 1,
                                 self.activation)
    
    def forward(self, x: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        # normalization
        if self.input_normalizer is not None:
            x = self.input_normalizer.encode(x)
        
        # add coordinate feature
        x = torch.cat((x, coord), dim=1)
        
        # encode
        x = x.permute(0, 2, 3, 1)
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)
        
        # add padding (left, right, top, bottom)
        x = F.pad(x, [0,self.padding,0,self.padding])

        # process
        for k, c in enumerate(zip(self.conv_layer, self.spconv_layer)):
            conv, spconv = c
            if k < self.layer_num-1:
                x = self.activation(conv(x) + spconv(x))
            else:
                x = conv(x) + spconv(x)

        # remove padding
        x = x[..., :-self.padding, :-self.padding]

        # decode
        x = self.decoder(x)

        # normalization
        if self.output_normalizer is not None:
            x = self.output_normalizer.decode(x)
        
        return x

class FixedConv2D(nn.Module):
    def __init__(self):
        super(FixedConv2D, self).__init__()
        self.input_channel = 1
        self.ou_channel = 3**2
        self.kernel_size = 3
        self.padding = 1

        self.conv = nn.Conv2d(self.input_channel, self.ou_channel,
            kernel_size=self.kernel_size, padding=self.padding)
        
        self.conv.weight.data = torch.zeros(self.ou_channel,1,3,3)
        self.conv.bias.data = torch.zeros(self.ou_channel)
        for i in range(3):
            for j in range(3):
                m = i*3 + j
                self.conv.weight.data[m,0,i,j] = 1.0
        
        for param in self.conv.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.conv(x)

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
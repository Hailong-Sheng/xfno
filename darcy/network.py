import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """ 2D Fourier layer
    Args:
        in_channel: number of input channels
        out_channel: number of output channels
        mode1 : number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
        mode2 : number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    """
    def __init__(self, in_channel: int, out_channel: int, mode1: int, mode2: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode1 = mode1
        self.mode2 = mode2

        self.scale = 1 / (in_channel * out_channel)
        self.weight1 = nn.Parameter(
            torch.empty(in_channel, out_channel, self.mode1, self.mode2, 2)
        )
        self.weight2 = nn.Parameter(
            torch.empty(in_channel, out_channel, self.mode1, self.mode2, 2)
        )
        self.reset_parameters()

    def compl_mul2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweight = torch.view_as_complex(weight)
        return torch.einsum("bixy,ioxy->boxy", input, cweight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Compute Fourier coeffcients up to factor of e^(-something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size,self.out_channel,x.size(-2),x.size(-1)//2+1,
                             dtype=torch.cfloat,device=x.device)
        out_ft[:,:,:self.mode1,:self.mode2] = self.compl_mul2d(
            x_ft[:,:,:self.mode1,:self.mode2], self.weight1)
        out_ft[:,:,-self.mode1:,:self.mode2] = self.compl_mul2d(
            x_ft[:,:,-self.mode1:,:self.mode2], self.weight2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weight with distribution scale*U(0,1)"""
        self.weight1.data = self.scale * torch.rand(self.weight1.data.shape)
        self.weight2.data = self.scale * torch.rand(self.weight2.data.shape)

class MLPLayer(nn.Module):
    """ channel-wise fully-connected like layer with 2d convolutions
    Args:
        in_channel: number of input channels
        out_channel: number of output channels
        mid_channel: number of middle channels
        actv: activation function
    """
    def __init__(self, in_channel, out_channel, mid_channel, actv=F.gelu):
        super(MLPLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mid_channel = mid_channel
        self.actv = actv

        self.mlp1 = nn.Conv2d(self.in_channel, self.mid_channel, 1)
        self.mlp2 = nn.Conv2d(self.mid_channel, self.out_channel, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.actv(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    """ 2D Fourier neural operator
    Args:
        in_channel: number of input channels
        out_channel: number of output channels
        width: number of hidden channels
        layer_num: number of spectral convolution layers
        mode_num: number of Fourier modes with learnable weight
        padding: padding size for FFT calculations
        actv: Activation function, by default Activation.GELU
        cord_feat: use coordinate meshgrid as additional input feature
    """
    def __init__(self, in_channel: int=1, out_channel: int=1, width: int=32, 
                 mode1: int=16, mode2: int=16, padding: int=8, layer_num: int=4, 
                 actv=F.gelu, input_scale=None, cord_feat: bool=True) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.width = width
        self.mode1 = mode1
        self.mode2 = mode2
        self.padding = padding
        self.layer_num = layer_num
        self.actv = actv
        self.input_scale = input_scale
        self.cord_feat = cord_feat    

        # add relative coordinate feature
        if self.cord_feat:
            self.in_channel += 2
        
        # encoder layer
        self.encoder = nn.Linear(self.in_channel, self.width)
        
        # convolutional layer
        self.spconv_layer = nn.ModuleList()
        self.conv_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            self.spconv_layer.append(
                SpectralConv2d(self.width, self.width, self.mode1, self.mode2)
            )
            self.conv_layer.append(nn.Conv2d(self.width, self.width, 1))

        # decoder layer
        self.decoder = MLPLayer(self.width, self.out_channel, self.width*4, self.actv)

    def meshgrid(self, shape, device: torch.device):
        batch_size, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.dim() == 4
        ), "Only 4D tensors [batch, in_channel, grid_x, grid_y] accepted for 2D FNO"

        # normalization
        x = (x - self.input_scale[0]) / self.input_scale[1]
        
        # add coordinate feature
        if self.cord_feat:
            cord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, cord_feat), dim=1)

        # encoder layer
        x = x.permute(0, 2, 3, 1)
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)
        
        # add padding (left, right, top, bottom)
        x = F.pad(x, [0,self.padding,0,self.padding])

        # spectral layers
        for k, c in enumerate(zip(self.conv_layer, self.spconv_layer)):
            conv, spconv = c
            if k < self.layer_num-1:
                x = self.actv(conv(x) + spconv(x))
            else:
                x = conv(x) + spconv(x)

        # remove padding
        x = x[..., :-self.padding, :-self.padding]

        # decorder layer
        x = self.decoder(x)
        return x

class FixedConv2D(nn.Module):
    def __init__(self):
        super(FixedConv2D, self).__init__()
        self.in_channel = 1
        self.ou_channel = 3**2
        self.kernel_size = 3
        self.padding = 1

        self.conv = nn.Conv2d(self.in_channel, self.ou_channel,
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.data import Data as PygData

from .layer import MLP, MLPConv2d, SpectralConv2d, GraphConv

class FCNet(torch.nn.Module):
    def __init__(self, layers, bounds):
        super(FCNet, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        self.bounds = bounds
        
        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)

    def forward(self, x):
        x_shape = x.shape
        x = x.permute(0,2,3,1).reshape(x_shape[0]*x_shape[2]*x_shape[3],x_shape[1])

        self.bounds = self.bounds.to(x.device)        
        h = (x-self.bounds[:,0])/(self.bounds[:,1]-self.bounds[:,0])
        x = 2*h - 1.0
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            tmp = torch.zeros(x.shape[0],self.layers[i+1]-self.layers[i]).to(x.device)
            x = h + torch.cat((x,tmp),1)
        x = self.fc[-1](x)
        return x.reshape(x_shape[0],x_shape[2],x_shape[3],self.layers[-1]).permute(0,3,1,2)

class FNO2d(nn.Module):
    """ 2D Fourier neural operator """
    def __init__(self, input_channel: int=1, output_channel: int=1, width: int=32, 
                 mode1_num: int=16, mode2_num: int=16, padding: int=8, layer_num: int=4, 
                 activation=nn.GELU(), input_scale=None, cord_feat: bool=True) -> None:
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
            cord_feat: whether to use coordinate as additional input feature
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
        self.input_scale = input_scale
        self.cord_feat = cord_feat    

        # add relative coordinate feature
        if self.cord_feat:
            self.input_channel += 2
        
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
        ), "Only 4D tensors [batch, input_channel, grid_x, grid_y] accepted for 2D FNO"

        # normalization
        if self.input_scale is not None:
            x = (x - self.input_scale[0]) / self.input_scale[1]
        
        # add coordinate feature
        if self.cord_feat:
            cord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, cord_feat), dim=1)

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

class GNO(torch.nn.Module):
    """ graph neural operator """
    def __init__(self, node_input_dim: int=1, edge_input_dim: int=1, node_output_dim: int=1,
                 node_hidden_dim: int=32, edge_hidden_dim: int=32, layer_num: int=1) -> None:
        """ initialization
        args:
            node_input_dim: dimension of node in input layer
            edge_input_dim: dimension of edge in input layer
            node_output_dim: dimension of node in output layer
            node_hidden_dim: dimension of node in hidden layer
            edge_hidden_dim: dimension of edge in hidden layer
            layer_num: number of hidden layer
        """
        super(GNO, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_output_dim = node_output_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.layer_num = layer_num

        self.node_proj_1 = torch.nn.Linear(self.node_input_dim, self.node_hidden_dim)
        self.edge_proj = MLP(self.edge_input_dim, self.node_hidden_dim**2,
                             self.edge_hidden_dim, layer_num=2)
        self.conv = GraphConv(self.node_hidden_dim, self.node_hidden_dim, aggr='mean')
        self.node_proj_2 = torch.nn.Linear(self.node_hidden_dim, self.node_output_dim)

    def forward(self, data: PygData) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_proj_1(x)
        edge_attr = self.edge_proj(edge_attr)
        for _ in range(self.layer_num):
            x = F.relu(self.conv(x, edge_index, edge_attr))
        x = self.node_proj_2(x)
        return x

class GINO(torch.nn.Module):
    """ geometry infromed neural operator """
    def __init__(self, encoder, processer, decoder, input_scale, c_size, r_size, nx):
        """ initialization
        args:
            encoder: encoder
            processer: processer
            decoder: decoder
            input_scale: scale of input
            c_size: size of the original mesh
            r_size: size of the transformed mesh
            nx: shape of the transformed mesh
        """
        super(GINO, self).__init__()
        self.encoder = encoder
        self.processer = processer
        self.decoder = decoder
        self.input_scale = input_scale
        self.c_size = c_size
        self.r_size = r_size
        self.nx = nx

    def forward(self, graph: PygData) -> torch.Tensor:
        graph = copy.deepcopy(graph)

        # normalize
        batch_size = int(graph.x.shape[0]/(self.c_size+self.r_size))
        graph.x = (graph.x-self.input_scale[0]) / self.input_scale[1]

        # encode
        x = self.encoder(graph)
        
        # process
        x = x.reshape(batch_size,self.c_size+self.r_size)
        x = x[:,self.c_size:]
        x = x.reshape(batch_size,1,self.nx[0],self.nx[1])
        
        x = self.processer(x)
        
        x = x.reshape(batch_size,self.r_size)
        x = torch.cat([torch.zeros(batch_size,self.c_size).to(x.device),
                       x],1)
        x = x.reshape(-1,1)
        
        # decode
        edge_index = graph.edge_index
        edge_index = torch.flip(edge_index, dims=[0])
        batch = PygData(x=x, y=graph.y, mask=graph.mask,
                        edge_index=edge_index, edge_attr=graph.edge_attr)
        x = self.decoder(batch)

        x = x.reshape(batch_size,self.c_size+self.r_size)
        x = x[:,:self.c_size]
        x = x.reshape(graph.y.shape)

        return x
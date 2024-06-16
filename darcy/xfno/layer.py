import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

class MLP(torch.nn.Module):
    """ multi-layer perceptron """
    def __init__(self, input_dim: int=1, output_dim: int=1, hidden_dim: int=32,
                 layer_num: int=1, activation=nn.ReLU()) -> None:
        """ initialization
        args:
            input_dim: dimension of input layer
            output_dim: dimension of output layer
            hidden_dim: dimension of hidden layer
            layer_num: number of hidden layer
            activation: activation function
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.activation = activation

        self.layers = nn.ModuleList()
        in_dim = self.input_dim
        for _ in range(self.layer_num):
            self.layers.append(nn.Linear(in_dim, self.hidden_dim))
            self.layers.append(self.activation)
            in_dim = self.hidden_dim
        self.layers.append(nn.Linear(in_dim, self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward propagation
        args:
            x: input tensor
        returns:
            output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x

class MLPConv2d(nn.Module):
    """ 2D multi-layer convolution (channel-wise) """
    def __init__(self, input_channel: int=32, output_channel: int=1, hidden_channel: int=32,
                 layer_num=1, activation=nn.GELU()) -> None:
        """ initialization
        args:
            input_channel: number of input channels
            output_channel: number of output channels
            hidden_channel: number of middle channels
            layer_num: number of hidden layer
            activation: activation function
        """
        super(MLPConv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channel = hidden_channel
        self.layer_num = layer_num
        self.activation = activation

        self.layers = nn.ModuleList()
        in_channel = self.input_channel
        for _ in range(self.layer_num):
            self.layers.append(nn.Conv2d(in_channel, self.hidden_channel, 1))
            self.layers.append(self.activation)
            in_channel = self.hidden_channel
        self.layers.append(nn.Conv2d(in_channel, self.output_channel, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward propagation
        args:
            x: input tensor
        returns:
            output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x

class SpectralConv2d(nn.Module):
    """ 2D spectral convolution layer """
    def __init__(self, input_channel: int=1, output_channel: int=1,
                 mode1_num: int=8, mode2_num: int=8) -> None:
        """ initialization
        args:
            input_channel: number of input channels
            output_channel: number of output channels
            mode1_num: number of Fourier modes in the 1th dimension
            mode2_num: number of Fourier modes in the 2th dimension
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.mode1_num = mode1_num
        self.mode2_num = mode2_num

        self.scale = 1 / (input_channel * output_channel)
        self.weight1 = nn.Parameter(
            torch.empty(input_channel, output_channel, self.mode1_num, self.mode2_num, 2)
        )
        self.weight2 = nn.Parameter(
            torch.empty(input_channel, output_channel, self.mode1_num, self.mode2_num, 2)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ reset spectral weight with distribution scale * U(0,1) """
        self.weight1.data = self.scale * torch.rand(self.weight1.data.shape)
        self.weight2.data = self.scale * torch.rand(self.weight2.data.shape)
    
    def compl_mul2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """ multiply complex tensors """
        # (batch, input_channel, x, y), (input_channel, output_channel, x, y) -> (batch, output_channel, x, y)
        cweight = torch.view_as_complex(weight)
        return torch.einsum("bixy,ioxy->boxy", input, cweight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward propagation
        args:
            x: input tensor
        returns:
            output tensor
        """
        batch_size = x.shape[0]

        # compute Fourier coeffcients up to factor of e^(-something constant)
        x_ft = torch.fft.rfft2(x)

        # multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size,self.output_channel,x.size(-2),x.size(-1)//2+1,
                             dtype=torch.cfloat,device=x.device)
        out_ft[:,:,:self.mode1_num,:self.mode2_num] = self.compl_mul2d(
            x_ft[:,:,:self.mode1_num,:self.mode2_num], self.weight1)
        out_ft[:,:,-self.mode1_num:,:self.mode2_num] = self.compl_mul2d(
            x_ft[:,:,-self.mode1_num:,:self.mode2_num], self.weight2)

        # return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class GraphConv(MessagePassing):
    """ graph convolution layer """
    def __init__(self, node_input_dim: int=1, node_output_dim: int=1, aggr: str='add', 
                 root_weight: bool=True, bias: bool=True) -> None:
        """ initialization
        args:
            node_input_dim: dimension of input node
            node_output_dim: dimension of output node
            edge_proj: edge projection
            aggr: aggregation manner
        """
        super(GraphConv, self).__init__(aggr=aggr)
        self.node_input_dim = node_input_dim
        self.node_output_dim = node_output_dim
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(self.node_input_dim, self.node_output_dim))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(self.node_output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        uniform(self.node_input_dim, self.root)
        uniform(self.node_input_dim, self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
                ) -> torch.Tensor:
        """ forward propagation
        args:
            x: node attribute
            edge_index: edge index
            edge_attr: edge attribute
        returns:
            output tensor
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        weight = edge_attr.view(-1, self.node_input_dim, self.node_output_dim)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    
    def __repr__(self) -> None:
        return '{}({}, {})'.format(self.__class__.__name__, self.node_input_dim,
                                   self.node_output_dim)
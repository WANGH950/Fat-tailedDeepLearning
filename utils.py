import torch.nn as nn
from collections import OrderedDict

class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, number_layer=3, alpha=1):
        super(FNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.number_layer = number_layer
        self.alpha = alpha
        layers = []
        if number_layer == 0:
            layers.append(('output_layer', nn.Linear(input_dim, output_dim)))
        else:
            layers.append(('hidden_layer_0', nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Softsign())))
            for i in range(1, number_layer-1):
                layers.append(('hidden_layer_'+str(i), nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Softsign())))
            layers.append(('output_layer', nn.Linear(hidden_dim, output_dim)))
        self.mlp = nn.Sequential(OrderedDict(layers))

    def forward(self, input):
        return self.mlp(input)

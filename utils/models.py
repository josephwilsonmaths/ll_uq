import torch

class mlp(torch.nn.Sequential):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", flatten=False, bias=True):
        super(mlp, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            act = torch.nn.ReLU
        elif activation == "tanh":
            act = torch.nn.Tanh
        else:
            raise ValueError('invalid activation')

        if flatten:
            self.add_module('flatten', torch.nn.Flatten())

        if len(hidden_sizes) == 0:
            # Linear Model
            self.add_module('lin_layer', torch.nn.Linear(self.input_size, self.output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', torch.nn.Linear(in_size, out_size, bias=bias))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', torch.nn.Linear(hidden_sizes[-1], self.output_size, bias=bias))
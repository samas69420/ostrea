import torch

class MLP(torch.nn.Module):

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 n_layers, 
                 hidden_dim,
                 activation_constructor,
                 device):

        super().__init__()

        layers = []

        for i in range(n_layers):

            if i == 0:
                layers.append(torch.nn.Linear(input_dim, hidden_dim))
                layers.append(activation_constructor())
            elif i == n_layers-1:
                layers.append(torch.nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_constructor())

        self.net = torch.nn.Sequential(*layers).to(device)

    def forward(self, x):
        
        return self.net(x)

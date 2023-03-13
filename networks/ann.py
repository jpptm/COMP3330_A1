import torch


class ANN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(ANN, self).__init__()
        # Build network using the Sequential class
        self.layers = [torch.nn.Linear(input_size, hidden_size[0]), activation]

        # Loop through the hidden layers and add them to the network, and then add the activation function in between
        for i in range(len(hidden_size) - 1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.layers.append(activation)

        self.layers.append(torch.nn.Linear(hidden_size[-1], output_size))
        self.layers.append(torch.nn.Softmax(dim=1))

        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.network(x)
        return out

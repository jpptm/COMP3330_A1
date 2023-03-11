import torch


# Cuda path C:\Users\User\AppData\Local\Temp\cuda

# # Read supplied data file then split to relevant values
# two_spiral_data = pandas.read_csv("spiralsdataset.csv", header=None).values
# x, y, group = two_spiral_data[:, 0], two_spiral_data[:, 1], two_spiral_data[:, 2]

# """plt.scatter(x, y, c=group)
# plt.grid("on")
# plt.show()"""

# print(torch.cuda.device_count())
# print(torch.__version__)

# print(torch.cuda.is_available())


# Things to try differently:
# Activation functions
# Number of layers
# Batch size
# Number of epochs
# Try activation functions in between layers and then just one at the end


# ANN class that inherits from pytorch - will be used to solve the 2 spiral problem
class ANN(torch.nn.Module):
    def __init__(self, device, input_size, hidden_sizes, output_size, activation):
        super().__init__()
        # CPU or CUDA compatible gpu
        self.device = device
        self.input_size = input_size
        # This allows us to have hidden layers that are of different sizes - hence an iterable
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        # Initialise layers
        self.layers = []

        # Add the input layer
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_sizes[0]))

        # Add the hidden layers and the chosen activation function
        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(
                torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
            )

            self.layers.append(self.activation)

        # Add the last hidden layer and the output layer
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1], self.output_size))

        # Use sequential to build the network
        self.network = torch.nn.Sequential(*self.layers)

    # Feed the input to the network using this member
    def forward(self, x):
        return self.network(x)


# Why we should insert an activation function in between the hidden layers https://www.youtube.com/watch?v=NkOv_k7r6no

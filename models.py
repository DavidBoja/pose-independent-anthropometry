
import torch.nn as nn
import torch.nn.functional as F

    
class SimpleMLP(nn.Module):
    def __init__(self, encoder_input_dim=368, output_dim=11, 
                 hidden_dim1=194, hidden_dim2=97):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(encoder_input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)

        return x
    
class BeefedMLP(nn.Module):
    def __init__(self, encoder_input_dim=368, output_dim=11, 
                 hidden_dim1=194, hidden_dim2=97):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(encoder_input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)

        return x
    

class EnhancedMLP(nn.Module):
    def __init__(self, encoder_input_dim=368, output_dim=11, 
                 hidden_dims=[256, 128, 64]):
        super(EnhancedMLP, self).__init__()

        layers = []
        input_dim = encoder_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Dropout layer
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
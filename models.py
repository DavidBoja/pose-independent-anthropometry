
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
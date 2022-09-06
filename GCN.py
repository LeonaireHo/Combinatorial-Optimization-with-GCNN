import torch
import torch.nn as nn

class simpleMLP(nn.Module):
    def __init__(self, input_size, out_size, activate=nn.ReLU()):
        super(simpleMLP, self).__init__()
        self.linear = nn.Linear(input_size, out_size)
        self.activate = activate

    def forward(self, x):
        x = self.linear(x)
        x = self.activate(x)
        return x


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, nb_MLP, layerNorm=False):
        super(GNN, self).__init__()
        self.size_mlp = nb_MLP
        self.mlp = nn.ModuleList(
            [simpleMLP(input_size, hidden_size)] + [simpleMLP(hidden_size, hidden_size)] + [
                simpleMLP(hidden_size * 2, hidden_size) for i in range(self.size_mlp - 2)]
        )
        self.layerNorm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for i in range(self.size_mlp)]
        )

    def forward(self, x, A):
        for i in range(self.size_mlp):
            x = self.mlp[i](x)
            y = A @ x
            y = self.layerNorm[i](y)
            if i != 0:
                x = torch.cat([y, x], 1)  # sum plutot que con
            else:
                x = y
        return x


class VariablePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, nb_MLP):
        super(VariablePredictor, self).__init__()
        self.GNN = GNN(input_size, hidden_size, nb_MLP)
        self.outlayer = simpleMLP(hidden_size * 2, 1, nn.Sigmoid())

    def forward(self, x, A):
        x = self.GNN(x, A)
        return self.outlayer(x)

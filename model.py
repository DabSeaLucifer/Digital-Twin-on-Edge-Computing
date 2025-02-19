import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, output_dim=3*32*32):  # CIFAR-10 has 3 color channels
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z).view(-1, 3, 32, 32)  # Output CIFAR-10-sized images

class MiniBatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features):
        super(MiniBatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        m = x @ self.T
        dist = torch.abs(m.unsqueeze(0) - m.unsqueeze(1)).sum(dim=2)
        return torch.cat([x, torch.exp(-dist).sum(dim=1)], dim=1)

class Discriminator(nn.Module):
    def __init__(self, input_dim=3*32*32):  # CIFAR-10 size
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.mbd = MiniBatchDiscrimination(256, 100)
        self.final = nn.Linear(256 + 100, 1)

    def forward(self, x):
        x = self.fc(x.view(x.size(0), -1))
        x = self.mbd(x)
        return self.final(x)
